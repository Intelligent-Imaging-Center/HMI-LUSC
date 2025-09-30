import torch
import torch.nn as nn

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #print(avg_out.size())
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HybridSN_BN_Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(HybridSN_BN_Attention, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels,out_channels=8,kernel_size=(7,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(5,3,3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.ca = ChannelAttention(32 * 18)
        self.sa = SpatialAttention()

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.conv2d_features(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

class CNN2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(CNN2D, self).__init__()
        self.ca = ChannelAttention(30)
        self.sa = SpatialAttention()

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.classifier = nn.Sequential(
            # nn.Linear(32 * 23 * 23, 256),
            nn.Linear(128 * 19 * 19, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])
        print(x.shape)
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.conv2d_features(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(CNN3D, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels,out_channels=8,kernel_size=(7,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(5,3,3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32,out_channels=64,kernel_size=(3,3,3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.ca = ChannelAttention(64 * 6)
        self.sa = SpatialAttention()

        self.classifier = nn.Sequential(
            # nn.Linear(576 * 19 * 19, 256),
            nn.Linear(64 * 6 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])

        x = self.ca(x) * x
        x = self.sa(x) * x

        # x = self.conv2d_features(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

# 打印网络结构  
# from torchinfo import summary

# # 网络放到GPU上
# net = HybridSN_BN_Attention()
# net = CNN2D()
# net = CNN3D()
# # 网络总结 
# summary(net, input_size=(100, 1, 30, 25, 25),col_names=['num_params','kernel_size','input_size','output_size'],
#         col_width=10,row_settings=['var_names'],depth=4)

