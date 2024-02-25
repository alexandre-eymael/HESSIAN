"""
AlexNet Implementation and Model Creation

This script implements the AlexNet architecture using PyTorch and provides functions to create
AlexNet models of different sizes (small, base, large). It also includes utilities to save and
load model checkpoints.

The AlexNet class defines the architecture of the model, including customizable feature layers
and classifier layers.

Functions:
- AlexNet_small(num_classes): Creates a small AlexNet model.
- AlexNet_base(num_classes): Creates a base AlexNet model.
- AlexNet_large(num_classes): Creates a large AlexNet model.
- create_AlexNet(num_classes, model_size): Creates an AlexNet model of a specified size.
- save_model(epoch, optimizer, loss, path): Saves a model checkpoint.
- load_model(path, device=None): Loads a model checkpoint.

Usage:
- You can run this script directly to create different-sized AlexNet models and print their number
  of parameters.

"""

import torch
from torch import nn

class AlexNet(nn.Module):
    """
    Implementation of AlexNet architecture.

    This class defines the AlexNet model architecture using customizable feature layers
    and classifier layers.

    Args:
        features (list): List of tuples defining the convolutional layers in the feature extractor.
        classifier (list): List of layers in the classifier.
        dropout (float): Dropout probability.

    Attributes:
        features (nn.ModuleList): List of convolutional layers.
        classifier (nn.ModuleList): List of classifier layers.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.

    """
    def __init__(self, features, classifier, dropout):
        super().__init__()

        self.features = nn.ModuleList()
        self._make_layers(features, True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.ModuleList()
        self._make_layers(classifier, False, dropout)

    def _make_layers(self, layers, is_conv, dropout=None):
        """
        Create layers based on the provided specifications.

        Args:
            layers (list): List of layer specifications.
            is_conv (bool): Indicates whether the layers are convolutional.
            dropout (float): Dropout probability.

        """
        in_channels = layers[0][0] if is_conv else layers[0]
        for v in layers:
            if is_conv:
                out_channels, kernel_size, stride, padding = v[1:]
                self.features.append(nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding))
                self.features.append(nn.ReLU(inplace=True))
                if layers.index(v) != len(layers) - 1:
                    self.features.append(nn.MaxPool2d(3, 2))
                in_channels = out_channels
            else:
                if dropout and layers.index(v) < len(layers) - 1:
                    self.classifier.append(nn.Dropout(dropout))
                self.classifier.append(nn.Linear(in_channels, v if isinstance(v, int) else v[0]))
                if v != layers[-1]:
                    self.classifier.append(nn.ReLU(inplace=True))
                in_channels = v if isinstance(v, int) else v[0]

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        return x

    @property
    def n_params(self):
        """
        Calculate the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, epoch, optimizer, loss, path):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch.
            optimizer (torch.optim.Optimizer): Optimizer state.
            loss (float): Current loss.
            path (str): Path to save the checkpoint.

        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_model(self, path, device=None):
        """
        Load model checkpoint.

        Args:
            path (str): Path to the checkpoint.
            device (torch.device): Device to load the model to.

        Returns:
            tuple: Tuple containing epoch, optimizer state, and loss.

        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['optimizer_state_dict'], checkpoint['loss']

def alexnet_small(num_classes):
    """
    Create a small AlexNet model (134,709 parameters).

    Args:
        num_classes (int): Number of output classes.

    Returns:
        AlexNet: Small AlexNet model instance.

    """
    return AlexNet(
        features=[(3, 4, 11, 4, 2), (4, 8, 5, 1, 2)],
        classifier=[8 * 6 * 6, 128, 64, num_classes],
        dropout=0.1
    )

def alexnet_base(num_classes):
    """
    Create a base AlexNet model (526,957 parameters).

    Args:
        num_classes (int): Number of output classes.

    Returns:
        AlexNet: Base AlexNet model instance.

    """
    return AlexNet(
        features=[(3, 8, 11, 4, 2), (8, 16, 5, 1, 2)],
        classifier=[16 * 6 * 6, 256, 128, num_classes],
        dropout=0.1
    )

def alexnet_large(num_classes):
    """
    Create a large AlexNet model (2,076,365 parameters).

    Args:
        num_classes (int): Number of output classes.

    Returns:
        AlexNet: Large AlexNet model instance.

    """
    return AlexNet(
        features=[(3, 8, 11, 4, 2), (8, 16, 5, 1, 2), (16, 32, 3, 1, 1)],
        classifier=[32 * 6 * 6, 512, 256, num_classes],
        dropout=0.2
    )

def create_alexnet(num_classes, model_size):
    """
    Create an AlexNet model of specified size.

    Args:
        num_classes (int): Number of output classes.
        model_size (str): Size of the model ('small', 'base', or 'large').

    Returns:
        AlexNet: AlexNet model instance.

    Raises:
        ValueError: If an invalid model size is provided.

    """
    match model_size:
        case 'small':
            return alexnet_small(num_classes)
        case 'base':
            return alexnet_base(num_classes)
        case 'large':
            return alexnet_large(num_classes)
        case _:
            raise ValueError('Invalid model size, please choose between small, base, and large')

if __name__ == '__main__':
    NUM_CLASSES = 61

    # SMALL MODEL
    small_model = alexnet_small(NUM_CLASSES)
    print('Number of parameters Small:', small_model.n_params)

    # BASE MODEL
    base_model = alexnet_base(NUM_CLASSES)
    print('Number of parameters Base:', base_model.n_params)

    # LARGE MODEL
    large_model = alexnet_large(NUM_CLASSES)
    print('Number of parameters Large:', large_model.n_params)
