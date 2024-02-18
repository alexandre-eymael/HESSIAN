import torch
from torch import nn

class AlexNet(nn.Module):
    
    def __init__(self, features, classifier, dropout):
        super(AlexNet, self).__init__()
        
        self.features = nn.ModuleList()
        self._make_layers(features, True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.ModuleList()
        self._make_layers(classifier, False, dropout)

    def _make_layers(self, layers, is_conv, dropout=None):
        in_channels = layers[0][0] if is_conv else layers[0]
        for v in layers:
            if is_conv:
                out_channels, kernel_size, stride, padding = v[1:]
                self.features.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
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
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        return x
    
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, epoch, optimizer, loss, path):
        # save the model as a checkpoint in case we want to continue training, we need to save the optimizer as well
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        
    def load_model(self, path):
        # load the model from a checkpoint
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['optimizer_state_dict'], checkpoint['loss']

def AlexNet_small(num_classes):
    return AlexNet(
        features=[(3, 4, 11, 4, 2), (4, 8, 5, 1, 2)],
        classifier=[8 * 6 * 6, 64, num_classes],
        dropout=0.1
    )

def AlexNet_base(num_classes):
    return AlexNet(
        features=[(3, 16, 11, 4, 2), (16, 32, 5, 1, 2), (32, 64, 3, 1, 1)],
        classifier=[64 * 6 * 6, 128, 128, num_classes],
        dropout=0.1
    )

def AlexNet_large(num_classes):
    return AlexNet(
        features=[(3, 64, 11, 4, 2), (64, 192, 5, 1, 2), (192, 384, 3, 1, 1)],
        classifier=[384 * 6 * 6, 2048, 2048, num_classes],
        dropout=0.2
    )

def create_AlexNet(num_classes, model_size):
    match model_size:
        case 'small':
            return AlexNet_small(num_classes)
        case 'base':
            return AlexNet_base(num_classes)
        case 'large':
            return AlexNet_large(num_classes)
        case _:
            raise ValueError('Invalid model size, please choose between small, base, and large')

if __name__ == '__main__':
    num_classes = 61
    
    # SMALL MODEL 
    small_model = AlexNet_small(num_classes)
    print('Number of parameters Small:', small_model.get_nb_params())
    
    # BASE MODEL
    base_model = AlexNet_base(num_classes)
    print('Number of parameters Base:', base_model.get_nb_params())
    
    # LARGE MODEL
    large_model = AlexNet_large(num_classes)
    print('Number of parameters Large:', large_model.get_nb_params())