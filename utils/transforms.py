import random
from typing import List, Sequence, Union, Tuple
import numbers
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, im1, im2=None, im3=None, im4=None):
        for t in self.transforms:
            im1, im2, im3, im4 = t(im1, im2, im3, im4)
        return im1, im2, im3, im4

class ToTensor(object):
    def __call__(self, im1, im2=None, im3=None, im4=None):
        im1 = F.to_tensor(im1)
        if im2 is not None:
            im2 = F.to_tensor(im2)
        if im3 is not None:
            im3 = F.to_tensor(im3)
        if im4 is not None:
            im4 = F.to_tensor(im4)
        return im1, im2, im3, im4

"""
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
"""
class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), interpolation=2):
        super().__init__()
        self.size = size
        self.scale = scale
        self.interpolation = interpolation
        self.antialias: Optional[Union[str, bool]] = "warn"

    def __call__(self, im1, im2=None, im3=None, im4=None):
        crop_params = T.RandomResizedCrop.get_params(im1, self.size, self.scale)

        im1 = F.resized_crop(im1, *crop_params, self.size, self.interpolation, antialias=self.antialias)
        if im2 is not None:
            im2 = F.resized_crop(im2, *crop_params, self.size, self.interpolation, antialias=self.antialias)
        if im3 is not None:
            im3 = F.resized_crop(im3, *crop_params, self.size, self.interpolation, antialias=self.antialias)
        if im4 is not None:
            im4 = F.resized_crop(im4, *crop_params, self.size, self.interpolation, antialias=self.antialias)
        return im1, im2, im3, im4

class RandomSelectFlip(object):
    def __init__(self, p=0.33):
        self.p = p

    def __call__(self, im1, im2=None, im3=None, im4=None):
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                return RandomHorizontalFlip()(im1, im2, im3, im4)
            else:
                return RandomVerticalFlip()(im1, im2, im3, im4)
        return im1, im2, im3, im4

class RandomHorizontalFlip(object):
    def __init__(self):
        super().__init__()

    def __call__(self, im1, im2=None, im3=None, im4=None):
        im1 = F.hflip(im1)
        if im2 is not None:
            im2 = F.hflip(im2)
        if im3 is not None:
            im3 = F.hflip(im3)
        if im4 is not None:
            im4 = F.hflip(im4)
        return im1, im2, im3, im4

class RandomVerticalFlip(object):
    def __init__(self):
        super().__init__()

    def __call__(self, im1, im2=None, im3=None, im4=None):
        im1 = F.vflip(im1)
        if im2 is not None:
            im2 = F.vflip(im2)
        if im3 is not None:
            im3 = F.vflip(im3)
        if im4 is not None:
            im4 = F.vflip(im4)
        return im1, im2, im3, im4

class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability."""
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, im1, im2=None, im3=None, im4=None):
        if torch.rand(1) < self.p:
            return im1, im2, im3, im4
        for t in self.transforms:
            im1, im2, im3, im4 = t(im1, im2, im3, im4)
        return im1, im2, im3, im4

class RandomRotation(object):
    def __init__(self, degrees, expand=False, center=None, fill=0):
        super().__init__()
        self.degrees = self._setup_angle(degrees, name="degrees", req_sizes=(2,))
        if center is not None:
            self._check_sequence_input(center, "center", req_sizes=(2,))
        self.center = center
        self.expand = expand
        self.fill = fill

    def _check_sequence_input(self, x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
        if not isinstance(x, Sequence):
            raise TypeError(f"{name} should be a sequence of length {msg}.")
        if len(x) not in req_sizes:
            raise ValueError(f"{name} should be a sequence of length {msg}.")

    def _setup_angle(self, x, name, req_sizes=(2,)):
        if isinstance(x, numbers.Number):
            if x < 0:
                raise ValueError(f"If {name} is a single number, it must be positive.")
            x = [-x, x]
        else:
            self._check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]

    def _check_sequence_input(self, x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
        if not isinstance(x, Sequence):
            raise TypeError(f"{name} should be a sequence of length {msg}.")
        if len(x) not in req_sizes:
            raise ValueError(f"{name} should be a sequence of length {msg}.")

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, im1, im2=None, im3=None, im4=None):
        angle = self.get_params(self.degrees)
        im1 = F.rotate(im1, angle=angle, expand=self.expand, center=None, fill=self.fill)
        if im2 is not None:
            im2 = F.rotate(im2, angle=angle, expand=self.expand, center=None, fill=self.fill)
        if im3 is not None:
            im3 = F.rotate(im3, angle=angle, expand=self.expand, center=None, fill=self.fill)
        if im4 is not None:
            im4 = F.rotate(im4, angle=angle, expand=self.expand, center=None, fill=self.fill)
        return im1, im2, im3, im4

