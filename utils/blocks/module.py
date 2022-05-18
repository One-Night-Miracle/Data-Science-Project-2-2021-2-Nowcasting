import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import make_layers

class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output

class Predictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = make_layers(params)

        ## for last layer only !
        self.interpolate = None
        for layer_name, v in params.items():
            if 'interpolate' in layer_name:
                self.interpolate = Interpolate(size=(v[0], v[1]))

    def forward(self, input):
        '''
        input: S*B*1*H*W
        :param input:
        :return:
        '''
        input = input.squeeze(2).permute((1, 0, 2, 3))
        output = self.model(input)
        if self.interpolate is not None:
            output = self.interpolate(output)
        return output.unsqueeze(2).permute((1, 0, 2, 3, 4))


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x