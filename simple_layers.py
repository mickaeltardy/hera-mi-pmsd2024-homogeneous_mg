import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGaussianNoise(nn.Module):
    def __init__(self, gaussian_noise_std, device):
        super(SimpleGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.device = device

    def forward(self, x):
        if not self.gaussian_noise_std:
            return x

        return self._add_gaussian_noise(x)

    def _add_gaussian_noise(self, image):
        noise = torch.normal(mean=0.0, std=self.gaussian_noise_std, size=image.size()).to(self.device)
        return image + noise

class SimpleConvLayer(nn.Module):
    def __init__(self, in_channels, number_of_filters=32, filter_size=(3, 3), stride=(1, 1)):
        super(SimpleConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )

    def forward(self, x):
        return F.relu(self.conv(x))
    
class SimpleMaxPool(nn.Module):
    def __init__(self):
        super(SimpleMaxPool, self).__init__()

    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return F.max_pool2d(x, kernel_size=stride, stride=stride, padding=padding)
    
class SimpleAvgPool(nn.Module):
    def __init__(self):
        super(SimpleAvgPool, self).__init__()

    def forward(self, x):
        n, c, h, w = x.size()
        return x.view(n, c, -1).mean(-1)
    
class SimplePad(nn.Module):
    def __init__(self):
        super(SimplePad, self).__init__()

    def forward(self, x, pad):
        return F.pad(x, pad)