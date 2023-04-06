import torch
import torch.nn.functional as F
from train_utils import ce_loss

import numpy as np

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')


class CSLLabeled(torch.nn.Module):
    def __init__(self, G, device='cuda:0'):
        super(CSLLabeled, self).__init__()
        self.device = torch.device(device)
        D = np.diag(np.diag(G))
        M = np.matmul(G, np.linalg.inv(D))

        self.G = torch.tensor(G, requires_grad=False).to(device)
        self.G = F.normalize(self.G, p=1, dim=1)

        self.M = torch.tensor(M, requires_grad=False).to(device)
        self.D = torch.tensor(D, requires_grad=False).to(device)

        self.adjustment = torch.log(torch.tensor(np.diag(D), requires_grad=False).to(self.device))

    def forward(self, inputs, targets, reduction='mean'):
        log_probs = F.log_softmax(inputs - self.adjustment , dim=1)
        weights = self.M[targets.cpu()].to(torch.device(self.device))
        product = weights * log_probs
        if reduction=='mean':
            return -1 * torch.mean(torch.sum(product, 1))
        else:
            return -1 * torch.sum(product, 1)


class CSLUnlabeled(torch.nn.Module):
    def __init__(self, G, kl_thresh=0.0, device='cuda:0'):
        super(CSLUnlabeled, self).__init__()
        self.device = torch.device(device)

        D = np.diag(np.diag(G))
        M = np.matmul(G, np.linalg.inv(D))

        self.G = torch.tensor(G, requires_grad=False).to(device)
        self.G = F.normalize(self.G, p=1, dim=1)

        self.M = torch.tensor(M, requires_grad=False).to(device)
        self.D = torch.tensor(D, requires_grad=False).to(device)

        self.adjustment = torch.log(torch.tensor(np.diag(D), requires_grad=False).to(self.device))
        self.kl_thresh = kl_thresh
    
    def mask(self, max_idx, logits):
        probs = F.softmax(logits, -1)
        target_distribution = self.G[max_idx]
        KL_div = torch.sum(target_distribution *\
                 torch.log(torch.div(target_distribution + 1e-7,\
                                        probs + 1e-7)), 1)
        return KL_div < self.kl_thresh
    

    def forward(self, logits_w, logits_s):
        if not torch.all(self.M >= 0):
            print("All elements of M matrix are not positive")
        
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask_ = self.mask(max_idx, logits_w)
        
        log_probs = F.log_softmax(logits_s - self.adjustment , dim=1)
        weights = self.M[max_idx.cpu()].to(torch.device(self.device))
        product = -1 * weights * log_probs
        return torch.mean(torch.sum(product, 1) * mask_), torch.mean(mask_.float())

