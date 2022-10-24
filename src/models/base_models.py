import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def requires_grad(self, req, pretrained):
        for name, param in self.named_parameters():
            # if I have to unfreeze and network was pretrained, keep common part frozen
            if req and pretrained and 'base.' in name:
                continue
            param.requires_grad = req

    def init_weights(self, nonlinearity, a=0):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if nonlinearity == 'relu':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif nonlinearity == 'leaky_relu':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu', a=a)
            elif isinstance(module, nn.GRU):
                for name, m in module.named_modules():
                    if 'bias' in name:
                        nn.init.constant_(m, 0.0)
                    elif 'weight' in name:
                        if nonlinearity == 'relu':
                            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        elif nonlinearity == 'leaky_relu':
                            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=a)