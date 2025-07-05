
import torch 
import torch.nn as nn

class residualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channel != out_channel
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x, fmap_dict=None, prefix=""):
        residual = self.shortcut(x) if self.use_shortcut else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out += residual

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out

        out = self.relu(out)

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.out"] = out
        return out
    

class audioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(audioCNN, self).__init__()
        self.init_conv = nn.Sequential( 
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.ModuleList([residualBlock(64, 64, stride=1) for _ in range(3)])
        self.layer2 = nn.ModuleList(
            [residualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [residualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)]
        )
        self.layer4 = nn.ModuleList(
            [residualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)]
        )

        self.adpt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.drop_out = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_map=False):
        feature_maps = {}
        if return_feature_map != False:
            feature_maps["conv1"] = self.conv1(x)
            x = feature_maps["conv1"]
        else:
            x = self.init_conv(x)

        def block_forward(blocks, x, name):
            for i, block in enumerate(blocks):
                if feature_maps is not None:
                    x = block(x, feature_maps, prefix=f"{name}.block{i}")
                else:
                    x = block(x)
            if feature_maps is not None:
                feature_maps[name] = x
            return x

        x = block_forward(self.layer1, x, "layer1") 
        x = block_forward(self.layer2, x, "layer2")
        x = block_forward(self.layer2, x, "layer3")
        x = block_forward(self.layer2, x, "layer4")

        x = self.adpt_pool(x)
        x = self.flatten(x)
        x = self.fc(self.drop_out(x))

        return x, feature_maps
