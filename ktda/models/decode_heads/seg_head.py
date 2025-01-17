from mmseg.registry import MODELS
from torch import nn as nn
from torch.nn import functional as F
import math
import torch
from torch import Tensor
from typing import Tuple, Type, Dict, List
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        do_attention1: bool = True,
        do_attention2: bool = True,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.do_attention1 = do_attention1
        self.do_attention2 = do_attention2
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        # if self.skip_first_layer_pe:
        #     queries = self.self_attn(q=queries, k=queries, v=queries)
        # else:
        #     q = queries + query_pe
        #     attn_out = self.self_attn(q=q, k=q, v=queries)
        #     queries = queries + attn_out
        # queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        # queries = queries + attn_out
        # queries = self.norm2(queries)

        queries = self.do_attn1(queries, keys, query_pe, key_pe)
        keys = self.do_attn2(queries, keys, query_pe, key_pe)

        # MLP block
        # mlp_out = self.mlp(queries)
        # queries = queries + mlp_out
        # queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        # keys = keys + attn_out
        # keys = self.norm4(keys)

        return queries, keys

    def do_attn1(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tensor:
        if not self.do_attention1:
            return queries
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        return queries

    def do_attn2(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tensor:
        if not self.do_attention2:
            return keys
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return keys


@MODELS.register_module()
class OursTwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        do_attn1: bool = True,
        do_attn2: bool = True,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    do_attention1=do_attn1,
                    do_attention2=do_attn2,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class PrefixModule(nn.Module):
    def __init__(
        self,
        num_classes=4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.classes = num_classes
    def forward(self, x):
        x = x[:, :self.classes]
        return x

@MODELS.register_module()
class OursDecoder(BaseDecodeHead):
    def __init__(
        self,
        *,
        transformer_dim: 256,
        transformer,
        activation: Type[nn.Module] = nn.GELU,
        token_lens=11,
        has_token=True,
        has_num_classes_fuse=True,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.has_token = has_token
        self.transformer_dim = transformer_dim
        self.transformer: OursTwoWayTransformer = MODELS.build(transformer)
        self.num_classes = kwargs["num_classes"]

        if has_token:
            self.token_list = nn.ModuleList()
            for _ in range(token_lens):
                self.token_list.append(nn.Embedding(1, transformer_dim))
        else:
            self.token_list = [[] for _ in range(token_lens)]

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
        #     ),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(
        #         transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        #     ),
        #     activation(),
        # )

        self.output_upscaling = nn.Sequential(
            nn.PixelShuffle(2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.PixelShuffle(2),
            activation(),
        )

        # self.output_hypernetwork_mlps = MLP(
        #     transformer_dim, transformer_dim, transformer_dim // 8, 3
        # )
        self.output_hypernetwork_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim // 16, 3
        )
        if has_num_classes_fuse:
            self.class_agg = nn.Conv2d(token_lens, self.num_classes, 1)
        else:
            self.class_agg = PrefixModule(self.num_classes)

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_embeddings - torch.Size([1, 256, 128, 128])
        image_pe - torch.Size([1, 256, 128, 128])
        """
        image_embeddings = inputs["image_embeddings"]
        image_pe = inputs["image_pe"]
        device = image_embeddings.device
        token_embedding = []
        for token in self.token_list:
            if self.has_token:
                token_embedding.append(token.weight)
            else:
                token_embedding.append(torch.zeros(
                    1, self.transformer_dim).requires_grad_(False).to(device))

        # print(f"token_embedding: {len}")
        output_tokens = torch.cat(
            token_embedding,
            dim=0,
        )

        tokens = output_tokens.unsqueeze(0).expand(
            image_embeddings.size(0), -1, -1
        )  # torch.Size([4, 11, 256])

        src = image_embeddings  # torch.Size([4, 256, 128, 128])
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape
        # Run the transformer
        # print(src.shape, pos_src.shape, tokens.shape)
        hs, src = self.transformer(
            src, pos_src, tokens
        )  # hs - torch.Size([BS, 11, 256]), src - torch.Size([BS, 16348, 256])
        mask_token_out = hs[:, :, :]

        # torch.Size([4, 256, 128, 128])
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(
            src
        )  # torch.Size([4, 32, 512, 512])
        hyper_in = self.output_hypernetwork_mlps(
            mask_token_out
        )  # torch.Size([1, 11, 32])
        # hyper_in = hyper_in[:, [0, 2], :]
        b, c, h, w = upscaled_embedding.shape
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        )  # torch.Size([1, 11, 512, 512])
        # torch.Size([bs, 1024, 32, 32])
        # conv,i-> torch.Size([bs,cls, 512, 512])
        seg_output = self.class_agg(seg_output)
        # return seg_output,hyper_in
        return {
            "seg_output": seg_output,
            "hyper_in": hyper_in
        }

    def cls_seg(self, feat):
        """Classify each pixel."""
        return feat["seg_output"]

    def predict_by_feat(
        self, seg_logits: Tensor, batch_img_metas: List[dict]
    ) -> Tensor:
        return seg_logits["seg_output"]

    def loss_by_feat(self, seg_logits: Dict[str, Tensor],
                     batch_data_samples) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_logits = seg_logits["seg_output"]
        return super().loss_by_feat(seg_logits, batch_data_samples)
