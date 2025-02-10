import torch
import torch.nn as nn
import torch.nn.functional as F
import collections as col
import sys
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

        self.fc1 = nn.Linear(128 * 17 * 9, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 4)

        self.gaussian_noise_layer = layers.SimpleGaussianNoise(gaussian_noise_std, device='cpu')
        self.dropout = nn.Dropout(p=1-nodropout_probability)

    def forward(self, x):
        x = self.gaussian_noise_layer(x)

        # first conv sequence
        x = self.conv_layer_dict["conv1"](x) # 650x600x1 -> 324x299x32
        # print(x.shape)
        # second conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 162x99x32
        x = self.conv_layer_dict["conv2a"](x) # 80x74x32
        x = self.conv_layer_dict["conv2b"](x) # 78x72x64
        # x = self.conv_layer_dict["conv2c"](x) # 214x164x64 -> 212x162x64
        # print(x.shape)

        # third conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 39x36x64
        x = self.conv_layer_dict["conv3a"](x) # 37x34x128
        x = self.conv_layer_dict["conv3b"](x) # 35x32x128
        # x = self.conv_layer_dict["conv3c"](x) # 102x77x128 -> 100x75x128
        # print(x.shape)

        # WARNING: This is technically correct, but not robust to model architecture changes.
        # x = self.all_views_pad(x, pad=(0, 1, 0, 0))

        # fourth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2)) # 17x16x128
        # x = self.conv_layer_dict["conv4a"](x) # 15x14x128
        # x = self.conv_layer_dict["conv4b"](x) # 13x12x128
        # x = self.conv_layer_dict["conv4c"](x) # 46x33x128 -> 44x31x128
        # print(x.shape)

        # fifth conv sequence
        # x = self.all_views_max_pool(x, stride=(2, 2)) # 6x6x128
        # x = self.conv_layer_dict["conv5a"](x) # 4x4x256
        # x = self.conv_layer_dict["conv5b"](x) # 2x2x256
        # x = self.conv_layer_dict["conv5c"](x) # 18x11x256 -> 16x9x256
        # x = self.all_views_avg_pool(x) 
        # x = self.all_views_max_pool(x, stride=(2, 2)) # 16x9x256 -> 8x4x256
        # print(x.shape)

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