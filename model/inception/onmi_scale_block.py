import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_lenght)
    mask = np.ones((number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        # do some init
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)  # mask some weight ? why

        self.padding = nn.ConstantPad1d((int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0)

        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=max_kernel_size)  # make a layer
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)  # init conv1d weight
        self.conv1d.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)  # init conv1d bias

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight * self.weight_mask
        # self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result


class full_scale(nn.Module):
    def __init__(self, layer_parameter_list, use_residual=True):
        super(full_scale, self).__init__()
        self.use_residual = use_residual
        layer_parameter_list = [[1, 10, 1], [1, 10, 3], [1, 10, 5], [1, 10, 8]]   # get the layer_parameter list

        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])   # build layer
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

        out_channels = int(np.cumsum(np.array(layer_parameter_list)[:, 1]))

        if self.use_residual == True:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=layer_parameter_list[0][0],
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0
                          ),
                nn.BatchNorm1d(num_features=out_channels)
            )

    def forward(self, x):

        if self.use_residual == True:
            x = x + self.shortcut(x)
        pass


class onmi_cnn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(onmi_cnn, self).__init__()

        layer_parameter_list1=[]
        self.layer1 = full_scale(layer_parameter_list1)  # this is 3 layer contain full-scale with residual
        layer_parameter_list2=[]
        self.layer2 = full_scale(layer_parameter_list2)
        layer_parameter_list3=[]
        self.layer3 = full_scale(layer_parameter_list3)

        # suppose we have all the kernel_size
        # every
