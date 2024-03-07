from collections import defaultdict
import torch
from torch import nn

from denoiser.config import instantiate


class BaseModel(nn.Module):
    def __init__(
            self, 
            losses = defaultdict(),
        ):
        super().__init__()
        self.losses = losses
        for loss_name, loss_fn in losses.items():
            self.losses[loss_name] = instantiate(loss_fn)
            

    def prepare_input(self, data):
        return data

    def forward(self, data):
        if self.training:
            data = self.prepare_input(data)
            out = self.forward_train(data)
        else:
            out = self.forward_test(data)

        return out

    def forward_train(self, data):
        out = defaultdict()
        for name, loss in self.losses.items():
            loss = None
        return loss


    def forward_test(self, data):
        pass

    def forward_test_tta(self, data):
        pass

