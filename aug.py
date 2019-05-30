#-*-coding:utf-8-*-
import mxnet as mx
import cv2
import math
import numpy as np
from mxnet import nd, gluon
from mxnet.image import Augmenter
import random
import sys
import json


class Augmenter_light(object):
    """Image Augmenter base class"""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in self._kwargs.items():
            if isinstance(v, nd.NDArray):
                v = v.asnumpy()
            if isinstance(v, np.ndarray):
                v = v.tolist()
                self._kwargs[k] = v

    def dumps(self):
        """Saves the Augmenter to string

        Returns
        -------
        str
            JSON formatted string that describes the Augmenter.
        """
        return json.dumps([self.__class__.__name__.lower(), self._kwargs])

    def __call__(self, src, index, area_random_num):
        """Abstract implementation body"""
        raise NotImplementedError("Must override implementation.")


def generate_template(img_shape):
    template_h = np.ones(img_shape)
    template_w = np.ones(img_shape)
    for k in range(template_h.shape[1]):
        template_h[:, k] = k * 1.0 / template_h.shape[1]
    for k in range(template_w.shape[0]):
        template_w[k, :] = k * 1.0 / template_w.shape[0]
    return template_h, template_w


def spatial_aug(orig_img, brightness_value, template_h, template_w):
    # area_random_num=random.randint(1,4)

    rand_theta = random.randint(1, 359)
    img = orig_img.asnumpy()
    # img = np.copy(orig_img)
    img = img.astype(np.float32)
    ## Rand Area Contrast
    h_rand = np.cos(rand_theta * 1.0 / 360 * 2.0 * 3.14159)  # Rand Select From[0, 360]
    w_rand = np.sin(rand_theta * 1.0 / 360 * 2.0 * 3.14159)
    # print('h_rand',h_rand)
    if h_rand < 0:
        new_template_h = (1 - template_h) * h_rand * h_rand
    else:
        new_template_h = template_h * h_rand * h_rand

    if w_rand < 0:
        new_template_w = (1 - template_w) * w_rand * w_rand
    else:
        new_template_w = template_w * w_rand * w_rand

    # brightness = config.TRAIN.aug_strategy.spatial
    brightness = brightness_value
    c = random.uniform(-brightness, brightness)
    img = img * (1 + (new_template_h + new_template_w) * c)
    # img = img * (1 + (new_template_h + new_template_w) )
    img = np.clip(img, 0, 255)
    # img =img[::-1]
    # np.transpose(img, (2, 0, 1))
    img = mx.nd.array(img, dtype=np.uint8)
    # img = img.transpose((2, 0, 1))  # hwc->chw
    return img


class SpatialLightAug(Augmenter):
    """Make SpatialLight templete.
    Parameters
    ----------
    brightness
    p : the possibility the img be aug
    """

    def __init__(self, brightness, possibility):
        super(SpatialLightAug, self).__init__(brightness=brightness)
        self.brightness = brightness
        self.p = possibility

    def __call__(self, src):
        """Augmenter body"""
        # return resize_short(src, self.size, self.interp)
        a = random.random()
        if a > self.p:
            return src
        else:
            # angle=random.randint(-self.maxangel,self.maxangel)
            template_h, template_w = generate_template(src.shape)
            # print('finish templete')
            return spatial_aug(src, self.brightness, template_h, template_w)


class RandomGammaAug(Augmenter):


    def __init__(self, gamma):
        super(RandomGammaAug, self).__init__()
        log_gamma_vari = np.log(gamma)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        self.gamma = np.exp(alpha)

    def _gamma_transform(self, src, gamma):
        gamma_table = [np.power(x / 255.0, self.gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        image = cv2.LUT(src, gamma_table)
        if image.shape != src.shape:
            image = np.expand_dims(image, axis=2)
        return image

    def __call__(self, src):
        src = src.asnumpy().astype("uint8")
        image = self._gamma_transform(src, self.gamma)
        image = nd.array(image)
        return image


class RandomPepperSaltNoiseAug(Augmenter):

    def __init__(self, SNR):
        super(RandomPepperSaltNoiseAug, self).__init__()
        self.snr = SNR

    def __call__(self, src):
        src = src.asnumpy()
        h, w, c = src.shape
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[self.snr, (1 - self.snr) / 2.0, (1 - self.snr) / 2.0])
        mask = np.repeat(mask, c, axis=2)
        src[mask == 1] == 255  # pepper
        src[mask == 2] = 0  # salt
        image = nd.array(src)
        return image


class GuassianBlurAug(Augmenter):


    def __init__(self, p=0.5):
        super(GuassianBlurAug, self).__init__()
        self.p = p

    def __call__(self, src):
        if self.p < np.random.rand():
            return src
        src = src.asnumpy()
        k = np.random.randint(3, 9)
        if k % 2 == 0:
            if np.random.rand() > 0.5:
                k += 1
            else:
                k -= 1
        s = np.random.uniform(1, 5)
        image = cv2.GaussianBlur(src, (k, k), s)
        if image.shape != src.shape:
            image = np.expand_dims(image, axis=2)
        return nd.array(image)


class GuassianNoiseAug(Augmenter):


    def __init__(self, mean, var):
        super(GuassianNoiseAug, self).__init__()
        self.mean = mean
        self.var = var

    def __call__(self, src):
        src = src.asnumpy()
        noise = np.random.normal(self.mean, self.var ** 0.5, src.shape)
        image = noise + src
        return nd.array(image)


class Bgr2yuvAug(Augmenter):
    def __init__(self):
        super(Bgr2yuvAug, self).__init__()

    def bgr2yuv444(self, img_bgr):
        img_h = img_bgr.shape[0]
        img_w = img_bgr.shape[1]
        uv_start_idx = img_h * img_w
        v_size = int(img_h * img_w / 4)

        def _trans(img_uv):
            img_uv = img_uv.reshape(int(math.ceil(img_h / 2.0)), int(math.ceil(img_w / 2.0)), 1)
            img_uv = np.repeat(img_uv, 2, axis=0)
            img_uv = np.repeat(img_uv, 2, axis=1)
            return img_uv

        img_yuv420sp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
        img_yuv420sp = img_yuv420sp.flatten()
        img_y = img_yuv420sp[:uv_start_idx].reshape((img_h, img_w, 1))
        img_u = img_yuv420sp[uv_start_idx:uv_start_idx + v_size]
        img_v = img_yuv420sp[uv_start_idx + v_size:uv_start_idx + 2 * v_size]
        img_u = _trans(img_u)
        img_v = _trans(img_v)
        img_yuv444 = np.concatenate((img_y, img_u, img_v), axis=2)
        return img_yuv444

    def __call__(self, src):
        src = src.asnumpy()
        src = src.astype("uint8")
        image = self.bgr2yuv444(src)
        return nd.array(image)



