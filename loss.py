import torch
import torch.nn as nn

##      交叉熵损失
class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss