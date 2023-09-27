import math
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(MLP, self).__init__()

        in_ch = inputs # Option.in_features
        out_ch = outputs # Option.out_features
        embed_chs = [256, 128, 64, 32] # Option.hidden_chs
        layers = []
        for h_ch in embed_chs:
            fc = nn.Linear(in_ch, h_ch)
            # self.init_fc(fc)
            layers.append(nn.Sequential(fc, nn.ReLU(), ))
            in_ch = h_ch
        out = nn.Linear(in_ch, out_ch)
        # self.init_fc(out)
        layers.append(out)
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def init_fc(fc):
        nn.init.kaiming_normal_(fc.weight.data, nonlinearity='relu')
        # fc.weight.data.normal_(0, 0.1)
        fc.bias.data.zero_()

    def forward(self, x):
        y = self.layers(x)
        return y