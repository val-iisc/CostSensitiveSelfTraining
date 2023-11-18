import torch
import torch.nn.functional as F

class CSLLoss(torch.nn.Module):
    def __init__(self, G):
        """
        Initialize the CSLLoss module.

        Args:
        - G (torch.Tensor): Matrix used to compute CSL weights.
        """
        super(CSLLoss, self).__init__()

        self.G = torch.tensor(G, requires_grad=False)
        self.D = torch.diag(torch.diag(self.G))
        self.M = torch.mm(self.G, torch.inverse(self.D))
        self.weights = torch.nn.functional.normalize(self.M, p=1, dim=1)

        self.adjustment = torch.log(torch.diag(self.G))

    def kl_divergence(self, p, q, epsilon=1e-7):
        """
        Calculate element-wise KL divergence between two NxD tensors representing categorical distributions.

        Args:
        - p (torch.Tensor): NxD tensor representing categorical distributions.
        - q (torch.Tensor): NxD tensor representing categorical distributions.
        - epsilon (float): Small constant to prevent log(0) error.

        Returns:
        - torch.Tensor: N-dimensional tensor containing the KL divergence between corresponding distributions.
        """
        p, q = p + epsilon, q + epsilon
        kl_values = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)
        return kl_values

    def forward(self, logits_weak, logits_strong,  KL_threshold=0.9):
        """
        Compute the CSL loss.

        Args:
        - logits_weak (torch.Tensor): Raw outputs from the model for weak aug sample.
        - 
        - targets (torch.Tensor): Ground truth labels.
        - KL_threshold (float): Threshold for KL divergence.

        Returns:
        - tuple: CSL loss and a mask indicating samples with high KL divergence.
        """
        targets = torch.argmax(logits_weak)
        logits_strong = logits_strong - self.adjustment
        target_distribution = self.G[targets]

        # Create a mask based on KL divergence threshold
        mask = (self.kl_divergence(target_distribution, F.softmax(logits_weak, dim=1)) > KL_threshold).float()

        # Compute the CSL loss using the weights and softmax of inputs
        loss = mask * torch.sum(self.weights * torch.log(F.softmax(logits_strong, dim=1)))
        loss = torch.mean(loss)

        return loss, mask
