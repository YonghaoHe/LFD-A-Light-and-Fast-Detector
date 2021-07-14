# -*- coding: utf-8 -*-
from albumentations import *
import cv2

__all__ = ['BGR2RGB']


# Color mode change (bgr -> rgb)

class BGR2RGB(ImageOnlyTransform):
    """Apply color mode conversion BGR2RGB
    by default, image loaded by cv2 is in BGR order
    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.):
        super(BGR2RGB, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **param):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_params(self):
        return {}
