import torch
import torch.nn.functional as F

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

# Example usage
if __name__ == "__main__":
    # Example tensor of shape (batch_size, num_classes, embedding_size)
    batch_size = 4
    num_classes = 15
    embedding_size = 16

    # Randomly initialize tensor
    tensor = torch.randn(batch_size, num_classes, embedding_size)

    # Compute the loss
    loss = cosine_similarity_loss(tensor)
    print("Loss:", loss.item())
