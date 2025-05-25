import random
import numbers
from collections import abc

import numpy as np
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

__all__ = [
    "Compose",
    "Resize",  # 尺寸缩减到对应size, 如果给定size为int,尺寸缩减到(size * height / width, size)
    "RandomScale",  # 尺寸随机缩放
    "RandomCrop",  # 随机裁剪,必要时可以进行padding
    "RandomFlip",  # 随机水平翻转
    "ColorJitter",  # 亮度,对比度,饱和度,色调
    "RandomRotation",  # 随机旋转
]

Sequence = abc.Sequence
Iterable = abc.Iterable


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        assert 'rgb' in sample.keys()
        assert 'label' in sample.keys()

        # BILINEAR
        # sample['rgb'] = F.resize(sample['rgb'], self.size, Image.BILINEAR)
        sample['rgb'] = F.resize(sample['rgb'], self.size, interpolation=InterpolationMode.BICUBIC)

        # NEAREST
        if 'thermal' in sample.keys():
            # sample['thermal'] = F.resize(sample['thermal'], self.size, Image.NEAREST)
        # sample['label'] = F.resize(sample['label'], self.size, Image.NEAREST)
            sample['thermal'] = F.resize(sample['thermal'], self.size, InterpolationMode.NEAREST)
        sample['label'] = F.resize(sample['label'], self.size, InterpolationMode.NEAREST)
        return sample


class RandomCrop(object):

    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            assert 'vis' in sample.keys()
            h, w = sample['rgb'].shape[1:]
            ch = int(h * (1 - self.crop_rate))
            cw = int(w * (1 - self.crop_rate))
            i = np.random.randint(0, h - ch + 1)
            j = np.random.randint(0, w - cw + 1)
            for key in sample.keys():
                sample[key] = F.crop(sample[key], i, j, ch, cw)

        return sample


class RandomFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            for key in sample.keys():
                sample[key] = F.hflip(sample[key])

        return sample


class RandomScale(object):
    def __init__(self, scale):
        assert isinstance(scale, Iterable) and len(scale) == 2
        assert 0 < scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, sample):
        assert 'rgb' in sample.keys()
        assert 'label' in sample.keys()

        w, h = sample['rgb'].size

        scale = random.uniform(self.scale[0], self.scale[1])
        size = (int(round(h * scale)), int(round(w * scale)))

        # # BILINEAR
        # # sample['rgb'] = F.resize(sample['rgb'], size, Image.BILINEAR)
        # sample['rgb'] = F.resize(sample['rgb'], size, InterpolationMode.BICUBIC)
        #
        # # NEAREST
        # if 'thermal' in sample.keys():
        #     # sample['thermal'] = F.resize(sample['thermal'], size, Image.NEAREST)
        # # sample['label'] = F.resize(sample['label'], size, Image.NEAREST)
        #     sample['thermal'] = F.resize(sample['thermal'], size, InterpolationMode.NEAREST)
        # sample['label'] = F.resize(sample['label'], size, InterpolationMode.NEAREST)

        for key in sample.keys():
            # BILINEAR for image
            if key in ['rgb']:
                sample[key] = F.resize(sample[key], size, InterpolationMode.BICUBIC)
            # NEAREST for thermal, label, bound
            else:
                sample[key] = F.resize(sample[key], size, InterpolationMode.NEAREST)
        return sample


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        assert 'rgb' in sample.keys()
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['rgb'] = transform(sample['rgb'])
        return sample


class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):

        return random.uniform(degrees[0], degrees[1])

    def __call__(self, sample):

        angle = self.get_params(self.degrees)
        for key in sample.keys():
            sample[key] = F.rotate(sample[key], angle, self.resample, self.expand, self.center)

        return sample


