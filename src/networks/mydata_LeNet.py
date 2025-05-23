import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

# 添加全局变量
REP_DIM = 512

class mydata_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = REP_DIM  # 使用全局变量
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(512, 512, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(512 * 8 * 8, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn2d6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class mydata_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = REP_DIM  # 使用全局变量
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(512, 512, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(512 * 8 * 8, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (8 * 8)), 512, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(512, 512, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d10 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d11 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d12 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv7 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv7.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)  # 32 32 512 512
        x = self.pool(F.leaky_relu(self.bn2d1(x)))  # 32 32 256 256
        x = self.conv2(x)  # 32 64 256 256
        x = self.pool(F.leaky_relu(self.bn2d2(x)))  # 32 64 128 128
        x = self.conv3(x)  # 32 128 128 128
        x = self.pool(F.leaky_relu(self.bn2d3(x)))  # 32 128 64 64
        x = self.conv4(x)  # 32 256 64 64
        x = self.pool(F.leaky_relu(self.bn2d4(x)))  # 32 256 32 32
        x = self.conv5(x)  # 32 512 32 32
        x = self.pool(F.leaky_relu(self.bn2d5(x)))  # 32 512 16 16
        x = self.conv6(x)  # 32 512 16 16
        x = self.pool(F.leaky_relu(self.bn2d6(x)))  # 32 512 8 8
        x = x.view(x.size(0), -1)  # 32 32768
        x = self.bn1d(self.fc1(x))  # 32 512
        
        # Decoder
        x = x.view(x.size(0), int(self.rep_dim / (8 * 8)), 8, 8)  # 32 8 8 8
        x = F.leaky_relu(x)
        x = self.deconv1(x)  # 32 512 8 8
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)  # 32 512 16 16
        x = self.deconv2(x)  # 32 512 16 16
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)  # 32 512 32 32
        x = self.deconv3(x)  # 32 256 32 32
        x = F.interpolate(F.leaky_relu(self.bn2d9(x)), scale_factor=2)  # 32 256 64 64
        x = self.deconv4(x)  # 32 128 64 64
        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), scale_factor=2)  # 32 128 128 128
        x = self.deconv5(x)  # 32 64 128 128
        x = F.interpolate(F.leaky_relu(self.bn2d11(x)), scale_factor=2)  # 32 64 256 256
        x = self.deconv6(x)  # 32 32 256 256
        x = F.interpolate(F.leaky_relu(self.bn2d12(x)), scale_factor=2)  # 32 32 512 512
        x = self.deconv7(x)  # 32 3 512 512
        x = torch.sigmoid(x)
        return x
