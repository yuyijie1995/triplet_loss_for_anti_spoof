import mxnet as mx
import numpy as np
import os
import cv2
import time
import random
import sys
import aug
import multiprocessing
from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, nd
from multiprocessing import Process, Lock


class readSingleID(dataset.Dataset):
    def __init__(self, list_name, color=1, length=3, group=1, random=False, is_train=True, transform=None):
        self._transform = transform
        self._list_name = list_name
        self._length = length
        self._is_train = is_train
        self._random = random
        self._color = color
        self._group = group
        if self._color == 0:  # read grayscale image if color is 0
            self._channel = 1
        else:
            self._channel = 3
        self._images = [line.strip("\n") for line in open(self._list_name)]
        self._num_image = len(self._images)

    def __getitem__(self, idx):
        # randomly choose 1 sequence
        label_array = nd.zeros(shape=(self._length * self._group,))
        image_array = nd.zeros(shape=(self._length * self._group, self._channel, 128, 128))  # caution

        if self._random:
            rand_list = random.sample(range(0, self._num_image), self._length)
            seq = 0
            for i in rand_list:
                image, label = self._images[i].split("\t")
                if os.path.isfile(image):
                    img = mx.image.imread(image, flag=self._color)
                else:
                    sys.exit(image + " cannot be found.")
                label = nd.array([int(label)])
                if self._transform is not None:
                    img = self._transform(img)
                img = img.expand_dims(axis=0)  # nd array
                label_array[seq] = label
                image_array[seq] = img
                seq += 1
        else:
            for g in range(self._group):
                start = random.randint(0, self._num_image - self._length)
                for i in range(start, start + self._length):
                    image, label = self._images[i].split("\t")
                    if os.path.isfile(image):
                        img = mx.image.imread(image,
                                              flag=self._color)  # rgb if self._color is 1, else grayscale is used
                    else:
                        sys.exit(image + " cannot be found.")
                    label = nd.array([int(label)])
                    if self._transform is not None:
                        img = self._transform(img)
                    label_array[g * self._length + i - start] = label
                    image_array[g * self._length + i - start] = img
        return image_array, label_array

    def __len__(self):
        if self._is_train:
            return 960
        else:
            return 960


class readSequentialImages2(dataset.Dataset):
    def __init__(self, list_path_living,list_path_spoof, color, num_seq=2, random=False, is_train=True, transform=None):
        self._transform = transform
        self._list_path_living = list_path_living
        self._list_path_spoof=list_path_spoof
        self._num_seq = num_seq
        self._num_seq_living = num_seq
        self._num_seq_spoof = num_seq-1
        self._is_train = is_train
        self._random = random
        self._color = color
        self._channel = 1 if color == "GRAY" else 3
        self._id = 0
        self._id_living=0
        self._id_spoof=0
        self._start = 0
        self._start_living = 0
        self._start_spoof = 0
        self._list = []
        self._list_living=[]
        self._list_spoof=[]
        self._frames = []
        self._frames_living = []
        self._frames_spoof = []
        self._len_living = 0
        self._len_spoof = 0
        self._readlist()
        self._seq_images_living = []
        self._seq_images_spoof=[]
        self._make_sequence()
        # assert self._len == len(self._seq_images), "[ERROR]Number of sequence is not correct."

    def _readlist(self):
        for lst in sorted(os.listdir(self._list_path_living)):
            images_living = [line.strip("\n") for line in open(os.path.join(self._list_path_living, lst))]
            self._list_living.append(images_living)
            self._frames_living.append(len(images_living))
            self._len_living += len(images_living) - self._num_seq_living + 1

        for lst in sorted(os.listdir(self._list_path_spoof)):
            images_spoof = [line.strip("\n") for line in open(os.path.join(self._list_path_spoof, lst))]
            self._list_spoof.append(images_spoof)
            self._frames_spoof.append(len(images_spoof))
            self._len_spoof += len(images_spoof) -self._num_seq_spoof + 1



    def _make_sequence(self):
        while True:
            if self._start_living + self._num_seq_living > self._frames[self._id_living]:
                if self._id_living < len(self._list_living) - 1:
                    self._id_living += 1
                else:
                    break
                    self._id_living = 0
                self._start_living = 0
            images = self._list_living[self._id_living]
            self._seq_images_living.append(images[self._start: self._start + self._num_seq_living])
            self._start_living += 1

        while True:
            if self._start_spoof + self._num_seq_spoof > self._frames[self._id_spoof]:
                if self._id_spoof < len(self._list_spoof) - 1:
                    self._id_spoof += 1
                else:
                    break
                    self._id_spoof = 0
                self._start_spoof = 0
            images = self._list_spoof[self._id_spoof]
            self._seq_images_spoof.append(images[self._start: self._start + self._num_seq_spoof])
            self._start_spoof += 1

    def __getitem__(self, idx):
        label_array = nd.zeros(shape=(self._num_seq_living+self._num_seq_spoof,))
        image_array = nd.zeros(shape=(self._num_seq_living+self._num_seq_spoof, self._channel, 128, 128))  # caution

        if self._random:
            seq = 0
            self._id_living = random.randint(0, len(self._list_living) - 1)
            self._id_spoof = random.randint(0, len(self._list_spoof) - 1)
            images_living = self._list_living[self._id_living]
            images_spoof = self._list_spoof[self._id_spoof]
            rand_list_living = random.sample(range(0, self._frames_living[self._id_living]), self._num_seq_living)
            rand_list_spoof = random.sample(range(0, self._frames_spoof[self._id_spoof]), self._num_seq_spoof)

            for i in rand_list_living:
                image, label = images_living[i].split("\t")
                if not os.path.isfile(image):
                    sys.exit("FILE: {} cannot be found.".format(image))
                # img = mx.image.imread(image, flag = self._color)
                if self._color == "GRAY":
                    img = nd.array(np.expand_dims(cv2.imread(image, 0), axis=2))
                else:
                    img = nd.array(cv2.imread(image))
                label = int(label)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1
            for i in rand_list_spoof:
                image, label = images_spoof[i].split("\t")
                if not os.path.isfile(image):
                    sys.exit("FILE: {} cannot be found.".format(image))
                # img = mx.image.imread(image, flag = self._color)
                if self._color == "GRAY":
                    img = nd.array(np.expand_dims(cv2.imread(image, 0), axis=2))
                else:
                    img = nd.array(cv2.imread(image))
                label = int(label)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1

        else:
            seq = 0
            seq_images_living = self._seq_images_living[idx]
            seq_images_spoof=self._seq_images_spoof[idx]
            for i in range(len(seq_images_living)):
                image, label = seq_images_living[i].split("\t")
                if not os.path.isfile(image):
                    sys.exit("FILE: {} cannot be found.".format(image))
                # img = mx.image.imread(image, flag = self._color)
                if self._color == "GRAY":
                    img = nd.array(np.expand_dims(cv2.imread(image, 0), axis=2))
                else:
                    img = nd.array(cv2.imread(image))
                label = int(label)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1

            for i in range(len(seq_images_spoof)):
                image,label=seq_images_spoof[i].split('\t')
                if not os.path.isfile(image):
                    sys.exit('FILE:{} cannot be found.'.format(image))
                if self._color == "GRAY":
                    img = nd.array(np.expand_dims(cv2.imread(image, 0), axis=2))
                else:
                    img = nd.array(cv2.imread(image))
                label = int(label)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1

        return image_array, label_array

    def __len__(self):
        if self._is_train:
            return 3200
        return self._len_living+self._len_spoof


class readSequentialImages(dataset.Dataset):
    def __init__(self, list_path, color=1, length=3, group=1, random=False, is_train=True, transform=None):
        self._transform = transform
        self._list_path = list_path
        self._length = length
        self._is_train = is_train
        self._random = random
        self._color = color
        self._group = group
        if self._color == 0:  # read grayscale image if color is 0
            self._channel = 1
        else:
            self._channel = 3

    def __getitem__(self, idx):
        # randomly choose 1 text file
        num_list = len(os.listdir(self._list_path))
        index = random.randint(0, num_list - 1)
        fname = os.path.join(self._list_path, sorted(os.listdir(self._list_path))[index])
        # randomly choose 1 sequence
        images = [line.strip("\n") for line in open(fname)]

        label_array = nd.zeros(shape=(self._length * self._group,))
        image_array = nd.zeros(shape=(self._length * self._group, self._channel, 128, 128))  # caution
        num_image = len(images)

        if self._random:
            rand_list = random.sample(range(0, num_image), self._length)
            seq = 0
            for i in rand_list:
                image, label = images[i].split("\t")
                if os.path.isfile(image):
                    img = mx.image.imread(image, flag=self._color)
                else:
                    sys.exit(image + " cannot be found.")
                label = nd.array([int(label)])
                if self._transform is not None:
                    img = self._transform(img)
                # img = img.expand_dims(axis = 0)# nd array
                label_array[seq] = label
                image_array[seq] = img
                seq += 1

        else:
            for g in range(self._group):
                start = random.randint(0, num_image - self._length)
                for i in range(start, start + self._length):
                    image, label = images[i].split("\t")
                    if os.path.isfile(image):
                        img = mx.image.imread(image,
                                              flag=self._color)  # rgb if self._color is 1, else grayscale is used
                    else:
                        sys.exit(image + " cannot be found.")
                    label = nd.array([int(label)])
                    if self._transform is not None:
                        img = self._transform(img)
                    label_array[g * self._length + i - start] = label
                    image_array[g * self._length + i - start] = img

        return image_array, label_array

    def __len__(self):
        if self._is_train:
            return 3200
        else:
            return 3200


class readSequentialImagesFromRec(dataset.Dataset):
    def __init__(self, rec, idx, num_seq, color, random=False, transform=None, is_train=True):
        self._transform = transform
        self._rec = rec
        self._idx = idx
        self._num_seq = num_seq
        self._random = random
        self._color = color
        self._is_train = is_train
        self._channel = 1 if color == "GRAY" else 3
        print("Loading {} images.".format(color))
        self._fork()
        self._id = 0
        self._start = 1
        self._len = 0
        self._id_range = []
        self._get_range()
        self._id_num = len(self._id_range)
        self._seq_range = []
        if self._random == False:
            self._make_sequence()
            assert self._len == len(self._seq_range), "[ERROR]Number of sequences is not correct."

    def __getstate__(self):
        """The dataset should be picklable when launching multiprocess workers"""
        state = self.__dict__.copy()
        state['_record'] = None
        return state

    def __setstate__(self, state):
        """after unpickled, you should invoke `_fork` operation."""
        self.__dict__ = state.copy()

    def _fork(self):
        self._record = mx.recordio.MXIndexedRecordIO(self._idx, self._rec, 'r')

    def _get_range(self):
        item0 = self._record.read_idx(0)
        header0, img0 = mx.recordio.unpack_img(item0)
        id_table = list(map(int, header0.label))
        for i in range(*id_table):
            item = self._record.read_idx(i)
            header, img = mx.recordio.unpack_img(item)
            self._id_range.append(header.label)
            self._len += header.label[1] - header.label[0] + 1 - self._num_seq
        self._len = int(self._len)

    def _make_sequence(self):
        while True:
            left, right = list(map(int, self._id_range[self._id]))
            if self._start + self._num_seq > right:
                if self._id < self._id_num - 1:
                    self._id += 1
                else:
                    break
                    self._id = 0
                left, right = list(map(int, self._id_range[self._id]))
                self._start = left
            start, end = int(self._start), int(self._start + self._num_seq)
            self._seq_range.append([start, end])
            self._start += 1

    def __getitem__(self, idx):
        label_array = nd.zeros(shape=(self._num_seq,))
        image_array = nd.zeros(shape=(self._num_seq, self._channel, 128, 128))

        if self._random:
            seq = 0
            self._id = random.randint(0, self._id_num - 1)
            left, right = list(map(int, self._id_range[self._id]))
            rand_list = random.sample(range(left, right), self._num_seq)
            for i in rand_list:
                item = self._record.read_idx(i)
                header, img = mx.recordio.unpack_img(item)
                label = int(header.label)
                if self._color == "GRAY":
                    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
                img = nd.array(img)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1
        else:
            start, end = self._seq_range[idx]
            seq = 0
            for i in range(start, end):
                item = self._record.read_idx(i)
                header, img = mx.recordio.unpack_img(item)
                label = int(header.label)
                if self._color == "GRAY":
                    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
                img = nd.array(img)
                if self._transform is not None:
                    img = self._transform(img)
                label_array[seq] = label
                image_array[seq] = img
                seq += 1

        return image_array, label_array

    def __len__(self):
        if self._is_train:
            return 3200
        else:
            return self._len


class trainTransform():
    def __init__(self, color):
        self._color = color
        self.resize = mx.image.ResizeAug(144)
        self.crop = mx.image.RandomCropAug((128, 128))
        self.crop_ratio = mx.image.RandomSizedCropAug((128, 128), 1.0, (0.8, 1.2))
        self.flip = mx.image.HorizontalFlipAug(p=0.5)
        self.cast = mx.image.CastAug(typ='float32')
        self.bright = mx.image.BrightnessJitterAug(0.1)
        self.contrast = mx.image.ContrastJitterAug(0.1)
        self.color = mx.image.ColorJitterAug(0.2, 0.2, 0.2)
        self.normalize = mx.image.ColorNormalizeAug(mean=nd.array([128, 128, 128]).reshape(3, 1, 1),
                                                    std=nd.array([128, 128, 128]).reshape(3, 1, 1))
        self.guassian_blur = aug.GuassianBlurAug(0.95)
        self.gamma_aug = aug.RandomGammaAug(0.8)
        self.guassian_noise = aug.GuassianNoiseAug(0, 5)
        self.spatial_light = aug.SpatialLightAug(0.6, 0.5)
        self.bgr2yuv = aug.Bgr2yuvAug()
        if color == "GRAY":
            self.rgb_mean = nd.array([0.5]).reshape(1, 1, 1)
            self.rgb_std = nd.array([0.5]).reshape(1, 1, 1)
        else:
            # self.rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            # self.rgb_std = nd.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            self.rgb_mean = nd.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
            self.rgb_std = nd.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def __call__(self, img):
        img = self.resize(img)
        img = self.crop(img)
        img = self.flip(img)
        img = self.gamma_aug(img)
        img = self.guassian_noise(img)
        img = self.guassian_blur(img)
        img = self.color(img)
        if self._color == "YUV":
            img = self.bgr2yuv(img)
        img = self.cast(img)
        img = img.transpose((2, 0, 1))  # (h,w,c) to (c,h,w)
        img = (img / 255.0 - self.rgb_mean) / self.rgb_std
        return img


class valTransform():
    def __init__(self, color):
        self._color = color
        self.resize = mx.image.ResizeAug(144)
        self.crop = mx.image.CenterCropAug((128, 128))
        self.cast = mx.image.CastAug(typ='float32')
        self.normalize = mx.image.ColorNormalizeAug(mean=nd.array([128, 128, 128]).reshape(3, 1, 1),
                                                    std=nd.array([128, 128, 128]).reshape(3, 1, 1))
        self.bgr2yuv = aug.Bgr2yuvAug()
        if color == "GRAY":
            self.rgb_mean = nd.array([0.5]).reshape(1, 1, 1)
            self.rgb_std = nd.array([0.5]).reshape(1, 1, 1)
        else:
            # self.rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            # self.rgb_std = nd.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            self.rgb_mean = nd.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
            self.rgb_std = nd.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def __call__(self, img):
        img = self.resize(img)
        img = self.crop(img)
        if self._color == "YUV":
            img = self.bgr2yuv(img)
        img = self.cast(img)
        img = img.transpose((2, 0, 1))
        img = (img / 255.0 - self.rgb_mean) / self.rgb_std
        return img


def loadData(list_path_living,list_path_spoof, batch_size, color, num_seq=3, is_train=True, random=False):
    # print("Batch size is %d, sequence length is %d" %(batch_size, sequence))
    if is_train:
        transformer = trainTransform(color)
    else:
        transformer = valTransform(color)
    image = readSequentialImages2(list_path_living=list_path_living,list_path_spoof=list_path_spoof, color=color, transform=transformer, num_seq=num_seq,
                                  is_train=is_train, random=random)
    data = gluon.data.DataLoader(image, batch_size=batch_size, shuffle=False, num_workers=1)
    return data


def loadRec(idx, rec, batch_size, color=None, random=False, num_seq=3, is_train=True):
    if is_train:
        transformer = trainTransform(color)
    else:
        transformer = valTransform(color)
    image = readSequentialImagesFromRec(idx=idx, rec=rec, transform=transformer, num_seq=num_seq, is_train=is_train,
                                        random=random, color=color)
    data = gluon.data.DataLoader(image, batch_size=batch_size, shuffle=False, num_workers=4)
    return data





if __name__ == "__main__":
    # transformer = trainTransform()
    # image_train = readSequentialImages(list_path = "/mnt/data-1/data/yuhao.dou/SiW/list/Train/", transform = transformer)
    # batch_size = 128
    # train_data = gluon.data.DataLoader(image_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    # train_data = loadRec('data/rec2.5/val_rect2.5.idx', 'data/rec2.5/val_rect2.5.rec', color = "GRAY", is_train = False, batch_size = 32, num_seq = 15, random = False)
    train_data = loadData("/mnt/data-3/data/yijie.yu/train_rect2.0_living",'/mnt/data-3/data/yijie.yu/train_rect2.0_spoof', 32, "GRAY", num_seq=2, is_train=False, random=True)
    idx = 1
    for data, label in train_data:
        idx += 1
        print(data.shape)
        print(label.shape)
        # print("Batch {}".format(idx))


