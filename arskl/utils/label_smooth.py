import torch


def gaussian_label_smoothing(targets, num_classes, smoothing=0.1, sigma=0.25):
    """
    Gaussian label smoothing for classification tasks.

    Args:
    - targets (torch.Tensor): The original target labels.
    - num_classes (int): Total number of classes.
    - smoothing (float): Smoothing factor.
    - sigma (float): Standard deviation for Gaussian distribution.

    Returns:
    - torch.Tensor: Smoothed labels.
    """
    targets = torch.unsqueeze(targets, 1)
    # Create a uniform distribution
    label_shape = torch.Size((targets.size(0), num_classes))
    smooth_labels = torch.full(label_shape, smoothing / (num_classes - 1))
    # Gaussian distribution centered around the target class
    x = torch.arange(0, num_classes, dtype=torch.float32)
    for idx, target in enumerate(targets):
        gaussian_weights = torch.exp(-(x - target.item()) ** 2 / (2 * sigma ** 2))
        smoothed_vector = gaussian_weights / gaussian_weights.sum()
        smooth_labels[idx] = smooth_labels[idx] * (1 - smoothing) + smoothed_vector * smoothing
    return smooth_labels


def laplace_label_smoothing(targets, num_classes, smoothing=0.1, b=0.1):
    """
    Laplace label smoothing for classification tasks.

    Args:
    - targets (torch.Tensor): The original target labels.
    - num_classes (int): Total number of classes.
    - smoothing (float): Smoothing factor.
    - b (float): Scale parameter for Laplace distribution.

    Returns:
    - torch.Tensor: Smoothed labels.
    """
    targets = torch.unsqueeze(targets, 1)
    label_shape = torch.Size((targets.size(0), num_classes))
    smooth_labels = torch.full(label_shape, smoothing / (num_classes - 1))
    x = torch.arange(0, num_classes, dtype=torch.float32)

    for idx, target in enumerate(targets):
        laplace_weights = torch.exp(-torch.abs(x - target.item()) / b)
        smoothed_vector = laplace_weights / laplace_weights.sum()
        smooth_labels[idx] = smooth_labels[idx] * (1 - smoothing) + smoothed_vector * smoothing
    return smooth_labels
