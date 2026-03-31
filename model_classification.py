import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)

    def forward(self, X):
        shortcut = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)

        X = shortcut + X
        X = F.relu(X)

        return X
    
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, filters, s=2):
        super(ConvolutionalBlock, self).__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=1, stride=s)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)

        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=F3, kernel_size=1, stride=s)
        self.shortcut_bn = nn.BatchNorm2d(F3)

    def forward(self, X):
        shortcut = self.shortcut_conv(X)
        shortcut = self.shortcut_bn(shortcut)

        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)

        X = shortcut + X
        X = F.relu(X)

        return X
    
class ResNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=5, stages=[3, 4, 6, 3], filters=[64, 256, 512, 1024, 2048]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=filters[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = self._make_stages(stages, filters)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[-1], num_classes)

    def _make_stages(self, stages, filters):
        layers = []
        in_channels = filters[0]
        for stage, filter in zip(stages, filters[1:]):
            filters_in_stage = [int(filter / 4), int(filter / 4), filter]
            for i in range(stage):
                if i == 0:
                    layers.append(ConvolutionalBlock(in_channels, filters_in_stage))
                else:
                    layers.append(IdentityBlock(filter, filters_in_stage))
                in_channels = filter
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_model(num_classes=5):
    """Helper function to create model instance"""
    model = ResNet(input_shape=(3, 224, 224), num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test the model
    model = get_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Print model information
    summary(model, input_size=(3, 224, 224))
    
    # Test forward pass
    B, C, H, W = 1, 3, 224, 224
    inputs = torch.randn(B, C, H, W).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"\nTest shapes:")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")