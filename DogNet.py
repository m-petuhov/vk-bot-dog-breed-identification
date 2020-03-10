import torch.nn as nn

from torchvision.models import googlenet


class DogNet(nn.Module):

    def __init__(self, out_classes):
        super(DogNet, self).__init__()

        self.base_model = googlenet(pretrained=True, aux_logits=True)
        self.base_model.aux1.fc2 = nn.Linear(1024, out_classes)
        self.base_model.aux2.fc2 = nn.Linear(1024, out_classes)
        self.base_model.fc = nn.Linear(1024, out_classes)

    def forward(self, x):
        return self.base_model._forward(x)
