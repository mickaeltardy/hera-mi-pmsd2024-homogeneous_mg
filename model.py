import torch
import torch.nn as nn
import torch.nn.functional as F
import collections as col

import simple_layers as layers


class BaselineBreastModel(nn.Module):
    def __init__(self, device, nodropout_probability=None, gaussian_noise_std=None):
        super(BaselineBreastModel, self).__init__()
        self.conv_layer_dict = col.OrderedDict()

        # first conv sequence
        self.conv_layer_dict["conv1"] = layers.SimpleConvLayer(1, number_of_filters=32, filter_size=(3, 3), stride=(2, 2))

        # second conv sequence
        self.conv_layer_dict["conv2a"] = layers.SimpleConvLayer(32, number_of_filters=64, filter_size=(3, 3), stride=(2, 2))
        self.conv_layer_dict["conv2b"] = layers.SimpleConvLayer(64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv2c"] = layers.SimpleConvLayer(64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1))

        # third conv sequence
        self.conv_layer_dict["conv3a"] = layers.SimpleConvLayer(64, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv3b"] = layers.SimpleConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv3c"] = layers.SimpleConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))

        # fourth conv sequence
        self.conv_layer_dict["conv4a"] = layers.SimpleConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv4b"] = layers.SimpleConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv4c"] = layers.SimpleConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))

        # fifth conv sequence
        self.conv_layer_dict["conv5a"] = layers.SimpleConvLayer(128, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv5b"] = layers.SimpleConvLayer(256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv5c"] = layers.SimpleConvLayer(256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self._conv_layer_ls = nn.ModuleList(self.conv_layer_dict.values())

        # Pool, flatten, and fully connected layers
        self.all_views_pad = layers.SimplePad()
        self.all_views_max_pool = layers.SimpleMaxPool()
        self.all_views_avg_pool = layers.SimpleAvgPool()

        self.fc1 = nn.Linear(256 * 8 * 4, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 4)

        self.gaussian_noise_layer = layers.SimpleGaussianNoise(gaussian_noise_std, device='cpu')
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.gaussian_noise_layer(x)

        # first conv sequence
        x = self.conv_layer_dict["conv1"](x) # 2600x2000x1 -> 1300x1000x32

        # second conv sequence
        x = self.all_views_max_pool(x, stride=(3, 3)) # 1300x1000x32 -> 433x333x32
        x = self.conv_layer_dict["conv2a"](x) # 433x333x32 -> 216x166x64
        x = self.conv_layer_dict["conv2b"](x) # 216x166x64 -> 214x164x64
        x = self.conv_layer_dict["conv2c"](x) # 214x164x64 -> 212x162x64

        # third conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 212x162x64 -> 106x81x64
        x = self.conv_layer_dict["conv3a"](x) # 106x81x64 -> 104x79x128
        x = self.conv_layer_dict["conv3b"](x) # 104x79x128 -> 102x77x128
        x = self.conv_layer_dict["conv3c"](x) # 102x77x128 -> 100x75x128

        # WARNING: This is technically correct, but not robust to model architecture changes.
        # x = self.all_views_pad(x, pad=(0, 1, 0, 0))

        # fourth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 100x75x128 -> 50x37x128
        x = self.conv_layer_dict["conv4a"](x) # 50x37x128 -> 48x35x128
        x = self.conv_layer_dict["conv4b"](x) # 48x35x128 -> 46x33x128
        x = self.conv_layer_dict["conv4c"](x) # 46x33x128 -> 44x31x128

        # fifth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 44x31x128 -> 22x15x128
        x = self.conv_layer_dict["conv5a"](x) # 22x15x128 -> 20x13x256
        x = self.conv_layer_dict["conv5b"](x) # 20x13x256 -> 18x11x256
        x = self.conv_layer_dict["conv5c"](x) # 18x11x256 -> 16x9x256
        # x = self.all_views_avg_pool(x) 
        x = self.all_views_max_pool(x, stride=(2, 2)) # 16x9x256 -> 8x4x256

        # Pool, flatten, and fully connected layers
        # x = torch.cat([
        #     x["L-CC"],
        #     x["R-CC"],
        #     x["L-MLO"],
        #     x["R-MLO"],
        # ], dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x