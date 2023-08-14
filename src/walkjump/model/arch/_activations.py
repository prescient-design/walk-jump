from torch import nn

ACTIVATION_STR_TO_TYPE = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
