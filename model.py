import torch.nn as nn
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
net = resnet50
fc_features = net.fc.in_features
net.fc = nn.Sequential(  # modify fully connected layer
    nn.Dropout(0.5),
    nn.Linear(fc_features, 196)
    )


# freeze layers
ct = 0  # children number
for child in net.children():
    ct += 1
    # print(ct)
    # print(child)
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
