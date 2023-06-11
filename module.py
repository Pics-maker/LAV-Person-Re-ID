import torch
import torch.nn as nn
from torchvision import models


class BaseResnet50(nn.Module):
    def __init__(self):
        super(BaseResnet50, self).__init__()

        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load('/mnt/data/module/resnet50-19c8e357.pth'))

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        # stem部分：conv + bn + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # block部分
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MyModule(nn.Module):

    def __init__(self, resnet50, GPU=False):
        super(MyModule, self).__init__()

        if GPU:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # 外观特征分支
        self.backbone1 = resnet50()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten1 = nn.Flatten()

        # 步态特征分支
        self.backbone2 = resnet50()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten2 = nn.Flatten()

        # 融合(AE)
        self.AE_encoder = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )
        self.AE_decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.Sigmoid()
        )

        # ID预测
        self.preID = nn.Linear(2048, 84)

    def forward(self, x: list):
        RGB = x[0]  # list
        Gait = x[1].to(self.device)
        appearance_weight = torch.t(torch.reshape(x[2], (1, x[2].size()[0]))).to(torch.float32).to(self.device)
        gait_weight = torch.t(torch.reshape(x[3], (1, x[3].size()[0]))).to(torch.float32).to(self.device)

        rgb1 = self.backbone1(RGB[0].to(self.device))
        rgb2 = self.backbone1(RGB[1].to(self.device))
        rgb3 = self.backbone1(RGB[2].to(self.device))
        rgb4 = self.backbone1(RGB[3].to(self.device))

        appearance_feature = (rgb1 + rgb2 + rgb3 + rgb4) / 4
        appearance_feature = self.maxpool1(appearance_feature)
        appearance_feature = self.flatten1(appearance_feature)

        gait_feature = self.backbone2(Gait)
        gait_feature = self.maxpool2(gait_feature)
        gait_feature = self.flatten2(gait_feature)

        fusion_feature = torch.cat((appearance_feature * appearance_weight, gait_feature * gait_weight), dim=1)
        encode_feature = self.AE_encoder(fusion_feature)
        decode_feature = self.AE_decoder(encode_feature)

        preID = self.preID(encode_feature)

        return encode_feature, preID, fusion_feature, decode_feature
