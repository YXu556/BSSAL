import math
import torch.nn as nn
from models.layers import BBB_Linear
from models.layers import ModuleWrapper


class BBBMLP(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, device="cuda"):
        super(BBBMLP, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        BBBLinear = BBB_Linear
        self.act = nn.ReLU

        self.fc0 = BBBLinear(inputs, 256, bias=True, priors=self.priors, device=device)
        self.act0 = self.act()

        self.fc1 = BBBLinear(256, 128, bias=True, priors=self.priors, device=device)
        self.act1 = self.act()

        self.fc2 = BBBLinear(128, 64, bias=True, priors=self.priors, device=device)
        self.act2 = self.act()

        self.fc3 = BBBLinear(64, 32, bias=True, priors=self.priors, device=device)
        self.act3 = self.act()

        self.fc4 = BBBLinear(32, outputs, bias=True, priors=self.priors, device=device)
