import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.double_conv(x)
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.double_conv(x)
        downsampled = self.pool(features)
        return features, downsampled

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DownsamplingBlock(in_channels, 64)
        self.down2 = DownsamplingBlock(64, 128)
        self.down3 = DownsamplingBlock(128, 256)
        self.down4 = DownsamplingBlock(256, 512)
        self.bottleneck = DoubleConvBlock(512, 1024)
        self.up1 = UpsamplingBlock(1024, 512)
        self.up2 = UpsamplingBlock(512, 256)
        self.up3 = UpsamplingBlock(256, 128)
        self.up4 = UpsamplingBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f1, d1 = self.down1(x)
        f2, d2 = self.down2(d1)
        f3, d3 = self.down3(d2)
        f4, d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)
        u1 = self.up1(bottleneck, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)
        outputs = self.final_conv(u4)
        outputs = self.softmax(outputs)
        return outputs

def get_model(num_classes=5):
    """Helper function to create model instance"""
    model = UNet(in_channels=3, out_channels=num_classes)
    return model

if __name__ == "__main__":
    # Test the model
    model = get_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    B, C, H, W = 1, 3, 224, 224
    inputs = torch.randn(B, C, H, W).to(device)
    
    # Print model summary
    summary(model, input_size=(C, H, W))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")