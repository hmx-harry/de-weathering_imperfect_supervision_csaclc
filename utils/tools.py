import re
import os
import torch.nn as nn

def sort_key(s):
    dir_name, file_name = os.path.split(s)
    file_no = int(re.findall(r'\d+', file_name)[-1])
    return file_no


def pad_input(input, under_scale = 4):
    height_org, width_org = input.shape[2], input.shape[3]
    if width_org % under_scale != 0 or height_org % under_scale != 0:
        width_res = width_org % under_scale
        height_res = height_org % under_scale
        if width_res != 0:
            width_div = under_scale - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = under_scale - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0
        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % under_scale == 0, 'width cant under_scaled by stride'
    assert height % under_scale == 0, 'height cant under_scaled by stride'
    return input, [pad_left, pad_right, pad_top, pad_bottom]

def pad_input_back(input, pad_list):
    pad_left, pad_right, pad_top, pad_bottom = pad_list
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

