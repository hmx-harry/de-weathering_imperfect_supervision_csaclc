from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate as rot
from scipy import signal
from skimage.transform import resize
import os
import torchvision
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import random
from glob import glob
import numpy as np
import cv2

"""
Snowmix and Rainmix data gugmentation, refer to https://visual.ee.ucla.edu/gt_rain.htm/
and http://visual.ee.ucla.edu/wstream.htm/ respectively.
"""

def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)

def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR)

def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR)

def roll_x(pil_img, level):
    """Roll an image sideways."""
    delta = int_parameter(sample_level(level), pil_img.width / 3)
    if np.random.random() > 0.5:
        delta = -delta
    xsize, ysize = pil_img.size
    delta = delta % xsize
    if delta == 0: return pil_img
    part1 = pil_img.crop((0, 0, delta, ysize))
    part2 = pil_img.crop((delta, 0, xsize, ysize))
    pil_img.paste(part1, (xsize - delta, 0, xsize, ysize))
    pil_img.paste(part2, (0, 0, xsize - delta, ysize))
    return pil_img

def roll_y(pil_img, level):
    """Roll an image sideways."""
    delta = int_parameter(sample_level(level), pil_img.width / 3)
    if np.random.random() > 0.5:
        delta = -delta
    xsize, ysize = pil_img.size
    delta = delta % ysize
    if delta == 0: return pil_img
    part1 = pil_img.crop((0, 0, xsize, delta))
    part2 = pil_img.crop((0, delta, xsize, ysize))
    pil_img.paste(part1, (0, ysize - delta, xsize, ysize))
    pil_img.paste(part2, (0, 0, xsize, ysize - delta))
    return pil_img

def zoom_x(pil_img, level):
    # zoom from .02 to 2.5
    rate = level
    zoom_img = pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (rate, 0, 0, 0, 1, 0),
        resample=Image.BILINEAR)
    # need to do reflect padding
    if rate > 1.0:
        orig_x, orig_y = pil_img.size
        new_x = int(orig_x / rate)
        zoom_img = np.array(zoom_img)
        zoom_img = np.pad(zoom_img[:, :new_x, :], ((0, 0), (0, orig_x - new_x), (0, 0)), 'wrap')
    return zoom_img

def zoom_y(pil_img, level):
    # zoom from .02 to 2.5
    rate = level
    zoom_img = pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, 0, 0, 0, rate, 0),
        resample=Image.BILINEAR)
    # need to do reflect padding
    if rate > 1.0:
        orig_x, orig_y = pil_img.size
        new_y = int(orig_y / rate)
        zoom_img = np.array(zoom_img)
        zoom_img = np.pad(zoom_img[:new_y, :, :], ((0, orig_y - new_y), (0, 0), (0, 0)), 'wrap')
    return zoom_img

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

augmentations = [
    rotate, shear_x, shear_y,
    zoom_x, zoom_y, roll_x, roll_y
]

def getRainLayer2(rand_id1, rand_id2, rain_mask_dir):
    path_img_rainlayer_src = os.path.join(rain_mask_dir, f'{rand_id1}-{rand_id2}.png')
    rainlayer_rand = cv2.imread(path_img_rainlayer_src).astype(np.float32) / 255.0
    rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
    return rainlayer_rand

def getRandRainLayer2(rain_mask_dir):
    rand_id1 = random.randint(1, 165)
    rand_id2 = random.randint(4, 8)
    rainlayer_rand = getRainLayer2(rand_id1, rand_id2, rain_mask_dir)
    return rainlayer_rand


def rain_mix_3f_mult(img_rainy, img_coll, rain_mask_dir, zoom_min=0.06, zoom_max=1.8):
    def obtain_mask(im):
        mask_rand = getRandRainLayer2(rain_mask_dir)
        mask = augment_and_mix(mask_rand, severity=3, width=3, depth=-1, zoom_min=zoom_min,
                                         zoom_max=zoom_max) * 1
        return im, mask

    def crop_map(im, mask, cropper_im, height, width):
        im = cropper_im(im)
        cropper_mask = rainmix_RandomCrop(mask.shape[:2], (height, width))
        mask = cropper_mask(mask)
        return im, mask

    def mix_mask(im, im_mask):
        im_ret = im + im_mask - im * im_mask
        im_ret = np.clip(im_ret, 0.0, 1.0)
        return (im_ret * 255).astype(np.uint8)


    img_aid1, img_aid2, img_gt = img_coll
    img_rainy = (img_rainy.astype(np.float32)) / 255.0
    img_aid1 = (img_aid1.astype(np.float32)) / 255.0
    img_aid2 = (img_aid2.astype(np.float32)) / 255.0
    img_gt = (img_gt.astype(np.float32)) / 255.0


    img_rainy, img_rainy_mask = obtain_mask(img_rainy)
    height = min(img_rainy.shape[0], img_rainy_mask.shape[0])
    width = min(img_rainy.shape[1], img_rainy_mask.shape[1])
    cropper_im = rainmix_RandomCrop(img_rainy.shape[:2], (height, width))
    img_rainy, img_rainy_mask = crop_map(img_rainy, img_rainy_mask, cropper_im, height, width)

    img_aid1, img_aid1_mask = obtain_mask(img_aid1)
    img_aid1, img_aid1_mask = crop_map(img_aid1, img_aid1_mask, cropper_im, height, width)
    img_aid2, img_aid2_mask = obtain_mask(img_aid2)
    img_aid2, img_aid2_mask = crop_map(img_aid2, img_aid2_mask, cropper_im, height, width)
    img_gt = cropper_im(img_gt)

    img_rainy_ret = mix_mask(img_rainy, img_rainy_mask)
    img_aid1_ret = mix_mask(img_aid1, img_aid1_mask)
    img_aid2_ret = mix_mask(img_aid2, img_aid2_mask)
    img_gt_ret = (img_gt * 255).astype(np.uint8)
    return img_rainy_ret, img_aid1_ret, img_aid2_ret, img_gt_ret

def rain_mix(img_rainy, img_gt, rain_mask_dir, zoom_min=0.06, zoom_max=1.8):
    img_rainy = (img_rainy.astype(np.float32)) / 255.0
    img_gt = (img_gt.astype(np.float32)) / 255.0
    img_rainy_ret = img_rainy
    img_gt_ret = img_gt

    rainlayer_rand2 = getRandRainLayer2(rain_mask_dir)
    rainlayer_aug2 = augment_and_mix(rainlayer_rand2, severity=3, width=3, depth=-1, zoom_min=zoom_min,
                                     zoom_max=zoom_max) * 1

    height = min(img_rainy.shape[0], rainlayer_aug2.shape[0])
    width = min(img_rainy.shape[1], rainlayer_aug2.shape[1])
    cropper = rainmix_RandomCrop(rainlayer_aug2.shape[:2], (height, width))
    rainlayer_aug2_crop = cropper(rainlayer_aug2)
    cropper = rainmix_RandomCrop(img_rainy.shape[:2], (height, width))
    img_rainy_ret = cropper(img_rainy_ret)
    img_gt_ret = cropper(img_gt_ret)

    img_rainy_ret = img_rainy_ret + rainlayer_aug2_crop - img_rainy_ret * rainlayer_aug2_crop
    img_rainy_ret = np.clip(img_rainy_ret, 0.0, 1.0)
    img_rainy_ret = (img_rainy_ret * 255).astype(np.uint8)
    img_gt_ret = (img_gt_ret * 255).astype(np.uint8)
    return img_rainy_ret, img_gt_ret,

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., zoom_min=0.06, zoom_max=1.8):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(2, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            if (op == zoom_x or op == zoom_y):
                rate = np.random.uniform(low=zoom_min, high=zoom_max)
                image_aug = apply_op(image_aug, op, rate)
            else:
                image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    max_ws = max(ws)
    rate = 1.0 / max_ws
    mixed = max((1 - m), 0.7) * image + max(m, rate * 0.5) * mix
    return mixed

def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.

class rainmix_RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


def snow_mix(im_in,
            SM_MEAN=.5,  # mean for gaussian noise
            SM_SD=1,  # standard deviation for gaussian noise
            SM_GAUSS_SD=3,  # gaussian blur standard deviation
            SM_SCALE_ARRAY=[.5, 1, 2, 3, 5],  # scale to determine motion blur kernel size
            SM_THRESH_RANGE=(.72, .78),  # threshold for gaussian noise map
            SM_ROTATE_RANGE=60,  # rotate range for motion blur kernel
            SM_NO_BLUR_FRAC=0,  # percent of time with no motion blur
            img_size=[256, 256],  # size of input image
            ):
    im_in = (im_in.astype(np.float32)) / 255.0

    #drawcv2('src', cv2.cvtColor(im_in, cv2.COLOR_RGB2BGR))
    img_size = [im_in.shape[0], im_in.shape[1]]
    # SNOW MIX
    final_mask = np.zeros((img_size[0], img_size[1]))
    threshold = random.uniform(SM_THRESH_RANGE[0], SM_THRESH_RANGE[1])
    base_angle = random.uniform(-1 * SM_ROTATE_RANGE, SM_ROTATE_RANGE)
    for scale in SM_SCALE_ARRAY:
        # Generate snow layer with gaussian map thresholding
        inv_scale = 1 / scale
        layer = np.random.normal(SM_MEAN, SM_SD, (int(img_size[0] * scale), int(img_size[1] * scale)))
        layer = gaussian_filter(layer, sigma=SM_GAUSS_SD)
        layer = layer > threshold
        layer = resize(layer, (img_size[0], img_size[1]))

        # motion blur
        kernel_size = random.randint(10, 15)
        angle = base_angle + random.uniform(-30, 30)  # angle for motion blur
        SM_KERNEL_SIZE = min(max(int(kernel_size * inv_scale), 3), 15)
        kernel_v = np.zeros((SM_KERNEL_SIZE, SM_KERNEL_SIZE))
        kernel_v[int((SM_KERNEL_SIZE - 1) / 2), :] = np.ones(SM_KERNEL_SIZE)
        kernel_v = rot(kernel_v, 90 - angle)
        if (scale > 4):
            kernel_v = gaussian_filter(kernel_v, sigma=1)
        elif (scale < 1):
            kernel_v = gaussian_filter(kernel_v, sigma=3)
        else:
            kernel_v = gaussian_filter(kernel_v, sigma=int(4 - scale))
        kernel_v *= 1 / np.sum(kernel_v)
        if random.random() > SM_NO_BLUR_FRAC:
            layer = signal.convolve2d(layer, kernel_v, boundary='symm', mode='same')

        # blend with final mask
        final_mask += layer - final_mask * layer

    # extend to 3 rgb channel dims
    final_mask = np.expand_dims(final_mask, 2)
    final_mask = np.tile(final_mask, (1, 1, 3))


    # show mask
    im_in = im_in + final_mask - im_in * final_mask
    im_in = np.clip(im_in, 0.0, 1.0)
    im_in = (im_in * 255).astype(np.uint8)
    return im_in

