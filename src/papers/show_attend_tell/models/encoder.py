import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Uses a pretrained ResNet to extract feature maps.
    Outputs a tensor of shape (batch_size, encoder_dim, num_pixels).
    num_pixels = H * W after the final convolutional layer.
    """

    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        )  # using pretrained
        modules = list(resnet.children())[:-2]  # to remove avgpool and fc layer
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )
        self.encoder_dim = 2048  # ResNet-101 final conv output channels

    def forward(self, images):
        """
        images: (batch_size, 3, height, width)
        returns: (batch_size, encoder_dim, num_pixels)
        """
        with torch.no_grad():
            features = self.resnet(images)  # (B, 2048, H', W')
        # Optional: reduce to a fixed spatial size
        features = self.adaptive_pool(
            features
        )  # (B, 2048, encoded_image_size, encoded_image_size)
        # Flatten spatial dimensions
        features = features.flatten(
            start_dim=2
        )  # (B, 2048, encoded_image_size * encoded_image_size)
        # (batch_size, encoder_dim, num_pixels)
        return features
