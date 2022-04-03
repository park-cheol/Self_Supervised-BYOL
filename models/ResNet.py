import torch
import torch.nn as nn
import torchvision.models as models
from models.MLP import MLP


class ResNet(nn.Module):

    def __init__(self, model_name, hidden_channels, proj_channels):
        super(ResNet, self).__init__()

        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif model_name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        else:
            raise Exception("BackBone Error")

        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # last fc제외
        self.projection = MLP(in_channels=resnet.fc.in_features, hidden_channels=hidden_channels,
                              proj_channels=proj_channels)

    def forward(self, inputs:torch.Tensor):
        y = self.encoder(inputs)
        y = y.view(y.size(0), y.size(1))
        z = self.projection(y)

        return z


if __name__ == "__main__":
    m = ResNet('resnet50', hidden_channels=2048*2, proj_channels=256).cuda()
    # m = torch.nn.Sequential(*list(m.children())[:-1]).cuda()
    print(m)