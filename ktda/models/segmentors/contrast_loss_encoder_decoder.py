import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (
    ConfigType,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
    add_prefix,
)
import torch
from mmseg.models import BaseSegmentor
from mmseg.models.segmentors import EncoderDecoder


def cosine_similarity_loss(tensor, alpha=0.5):
    """
    Compute the loss based on cosine similarity for a tensor of shape
    (batch_size, num_classes, embedding_size).

    Args:
        tensor (torch.Tensor): Input tensor with shape (batch_size, num_classes, embedding_size).
        alpha (float): Hyperparameter to adjust margin for inter-class similarity. Default is 0.5.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    _, num_classes, _ = tensor.shape

    # Normalize the embeddings to compute cosine similarity
    tensor = F.normalize(tensor, p=2, dim=-1)

    # Compute pairwise cosine similarity
    cosine_sim = torch.matmul(tensor, tensor.transpose(1, 2))  # Shape: (batch_size, num_classes, num_classes)

    # Loss for intra-class similarity (maximize similarity within the same class)
    intra_class_loss = 0
    for i in range(num_classes):
        intra_class_sim = cosine_sim[:, i, i]  # Diagonal elements for class i (shape: batch_size)
        intra_class_loss += (1 - intra_class_sim).mean()

    # Loss for inter-class similarity (minimize similarity between different classes)
    inter_class_loss = 0
    for i in range(num_classes):
        inter_class_mask = torch.ones(num_classes, device=tensor.device, dtype=torch.bool)
        inter_class_mask[i] = 0  # Mask to exclude the diagonal element for class i
        inter_class_sim = cosine_sim[:, i, :][:, inter_class_mask]  # Off-diagonal elements for class i
        inter_class_loss += F.relu(inter_class_sim - alpha).mean()

    # Combine the two losses
    loss = intra_class_loss + inter_class_loss

    return loss

@MODELS.register_module()
class ContrastLossEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors."""

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        has_contrast_loss: bool = False,
        lambda_contrast: float = 0.2,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.has_contrast_loss = has_contrast_loss
        self.lambda_contrast = lambda_contrast
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        
        if self.has_contrast_loss:
            hyper_in = self.decode_head.forward(x)["hyper_in"]
            loss_contrast = cosine_similarity_loss(hyper_in)
            loss_contrast = self.lambda_contrast * loss_contrast
            losses.update({"decode_contrast_loss":loss_contrast})

        return losses
