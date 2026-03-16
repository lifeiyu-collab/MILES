import torch
import torch.nn.functional as F


class EntropyMinimization(torch.nn.Module):
    """Entropy Minimization loss

    Arguments:
        t : temperature
    """

    def __init__(self, t=1.):
        super(EntropyMinimization, self).__init__()
        self.t = t

    def forward(self, lbl, pred):
        """Compute loss.

        Arguments:
            lbl (torch.tensor:float): predictions, not confidence, not label.
            pred (torch.tensor:float): predictions.

        Returns:
            loss (torch.tensor:float): entropy minimization loss

        """
        loss = - F.softmax(lbl / self.t, dim=-1) * F.log_softmax(pred / self.t, dim=-1)
        return loss
