import torch
import torch.nn as nn
from monai.networks.nets import ResNet # MONAI is the standard for 3D Med-AI in 2026

class MRIQuality3D(nn.Module):
    def __init__(self):
        super(MRIQuality3D, self).__init__()
        # Using MONAI's ResNet implementation for 3D
        self.model = ResNet(
            block="basic",
            layers=[2, 2, 2, 2], # ResNet-18 configuration
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=1,
            num_classes=1,       # Binary output
            spatial_dims=3       # This activates 3D convolutions
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Depth, Height, Width)
        # e.g., (1, 1, 160, 256, 256)
        logits = self.model(x)
        return self.sigmoid(logits)