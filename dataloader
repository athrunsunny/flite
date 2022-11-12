import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import torch_distributed_zero_first, LOGGER, is_colab, is_kaggle,clean_str

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadFaceImagesAndLabels(path, imgsz, batch_size,
                                          augment=augment,  # augment images
                                          hyp=hyp,  # augmentation hyperparameters
                                          rect=rect,  # rectangular training
                                          cache_images=cache,
                                          single_cls=opt.single_cls,
                                          stride=int(stride),
                                          pad=pad,
                                          image_weights=image_weights,
                                          prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadFaceImagesAndLabels.collate_fn4 if quad else LoadFaceImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadFaceImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', rgb_means=(104, 117, 123)):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.rgb_means = rgb_means

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n{help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=prefix + desc, total=n, initial=n)
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        pbar = tqdm(zip(self.img_files, self.label_files), desc=desc, total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                        # print("ol",l)
                    l = self.clap_out_of_bound(shape, l)
                    # print('h',l)

                    # if len(l):
                    #     assert l.shape[1] == 15, 'labels require 15 columns each'
                    #     assert (l >= -1).all(), 'negative labels'
                    #     assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    #     assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    if not len(l):
                        ne += 1  # label empty
                        l = np.zeros((0, 15), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 15), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))
                # print(l.tolist())
                # print(shape)

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f"{prefix}New cache created: {path}")
        return x

    def clap_out_of_bound(self, shape, labels):
        # Prevention of cross the border
        cls = labels[:, :1] + 1
        bbox = labels[:, 1:5]
        # print("bbox",bbox)
        lmk = labels[:, 5:]
        # print("lmk",lmk)
        obbox = reconvert_np(shape, bbox)
        olmk = reconvert_lmk_np(shape, lmk)
        l1 = lmk[:, :1] < 0
        cls[l1] = -1

        slmk = torch.from_numpy(olmk)
        slmk = torch.clamp(slmk, max=shape[0]).numpy()
        sbbox = torch.from_numpy(obbox)
        sbbox = torch.clamp(sbbox, max=shape[0]).numpy()

        # lmk_ = convert_lmk(shape, slmk)
        # bbox_ = convert_scale(shape,sbbox)
        # bbox_ = convert(shape, sbbox)
        # lmk_[lmk_[:,:]<0] =-1
        # print("bbox_",bbox_)
        # print("lmk_",lmk_)
        return np.hstack((cls, sbbox, slmk))

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]
        image, (h0, w0), (h, w) = load_image(self, index)
        targets = self.labels[index]

        boxes = targets[:, 1:5].copy()
        labels = targets[:, 0].copy()
        landm = targets[:, 5:].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_size)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_size, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return torch.from_numpy(image_t), targets_t, self.img_files[index], (h0, w0)

    def __getitem1__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic_face(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic_face(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

                # labels[:, 5] = ratio[0] * w * x[:, 5] + pad[0]  # pad width
                labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 5] + pad[0]) + (
                        np.array(x[:, 5] > 0, dtype=np.int32) - 1)
                labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 6] + pad[1]) + (
                        np.array(x[:, 6] > 0, dtype=np.int32) - 1)
                labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 7] + pad[0]) + (
                        np.array(x[:, 7] > 0, dtype=np.int32) - 1)
                labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 8] + pad[1]) + (
                        np.array(x[:, 8] > 0, dtype=np.int32) - 1)
                labels[:, 9] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 9] + pad[0]) + (
                        np.array(x[:, 9] > 0, dtype=np.int32) - 1)
                labels[:, 10] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 10] + pad[1]) + (
                        np.array(x[:, 10] > 0, dtype=np.int32) - 1)
                labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 11] + pad[0]) + (
                        np.array(x[:, 11] > 0, dtype=np.int32) - 1)
                labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 12] + pad[1]) + (
                        np.array(x[:, 12] > 0, dtype=np.int32) - 1)
                labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 13] + pad[0]) + (
                        np.array(x[:, 13] > 0, dtype=np.int32) - 1)
                labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 14] + pad[1]) + (
                        np.array(x[:, 14] > 0, dtype=np.int32) - 1)

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            labels[:, [5, 7, 9, 11, 13]] /= img.shape[1]  # normalized landmark x 0-1
            labels[:, [5, 7, 9, 11, 13]] = np.where(labels[:, [5, 7, 9, 11, 13]] < 0, -1, labels[:, [5, 7, 9, 11, 13]])
            labels[:, [6, 8, 10, 12, 14]] /= img.shape[0]  # normalized landmark y 0-1
            labels[:, [6, 8, 10, 12, 14]] = np.where(labels[:, [6, 8, 10, 12, 14]] < 0, -1,
                                                     labels[:, [6, 8, 10, 12, 14]])

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

                    labels[:, 6] = np.where(labels[:, 6] < 0, -1, 1 - labels[:, 6])
                    labels[:, 8] = np.where(labels[:, 8] < 0, -1, 1 - labels[:, 8])
                    labels[:, 10] = np.where(labels[:, 10] < 0, -1, 1 - labels[:, 10])
                    labels[:, 12] = np.where(labels[:, 12] < 0, -1, 1 - labels[:, 12])
                    labels[:, 14] = np.where(labels[:, 14] < 0, -1, 1 - labels[:, 14])

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

                    labels[:, 5] = np.where(labels[:, 5] < 0, -1, 1 - labels[:, 5])
                    labels[:, 7] = np.where(labels[:, 7] < 0, -1, 1 - labels[:, 7])
                    labels[:, 9] = np.where(labels[:, 9] < 0, -1, 1 - labels[:, 9])
                    labels[:, 11] = np.where(labels[:, 11] < 0, -1, 1 - labels[:, 11])
                    labels[:, 13] = np.where(labels[:, 13] < 0, -1, 1 - labels[:, 13])

                    # 左右镜像的时候，左眼、右眼，　左嘴角、右嘴角无法区分, 应该交换位置，便于网络学习
                    eye_left = np.copy(labels[:, [5, 6]])
                    mouth_left = np.copy(labels[:, [11, 12]])
                    labels[:, [5, 6]] = labels[:, [7, 8]]
                    labels[:, [7, 8]] = eye_left
                    labels[:, [11, 12]] = labels[:, [13, 14]]
                    labels[:, [13, 14]] = mouth_left

        # 转回原坐标，扣除非正数坐标

        labels_out = torch.zeros((nL, 16))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
            # showlabels(img, labels[:, 1:5], labels[:, 5:15])
            labels_out[:, 1] += 1

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # print(index, '   --- labels_out: ', labels_out)
        # if nL:
        # print( ' : landmarks : ', torch.max(labels_out[:, 5:15]), '  ---   ', torch.min(labels_out[:, 5:15]))
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        # img, label, path, shapes = zip(*batch)  # transposed
        # for i, l in enumerate(label):
        #     l[:, 0] = i  # add target image index for build_targets()
        # return torch.stack(img, 0), torch.cat(label, 0), path, shapes

        targets = []
        imgs = []
        for _, sample in enumerate(batch):
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    annos = torch.from_numpy(tup).float()
                    targets.append(annos)

        return torch.stack(imgs, 0), targets, _, _

    # @staticmethod
    # def collate_fn4(batch):
    #     im, label, path, shapes = zip(*batch)  # transposed
    #     n = len(shapes) // 4
    #     im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
    #
    #     ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    #     wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    #     s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
    #     for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
    #         i *= 4
    #         if random.random() < 0.5:
    #             im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
    #                                 align_corners=False)[0].type(im[i].type())
    #             lb = label[i]
    #         else:
    #             im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
    #             lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
    #         im4.append(im1)
    #         label4.append(lb)
    #
    #     for i, lb in enumerate(label4):
    #         lb[:, 0] = i  # add target image index for build_targets()
    #
    #     return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


def showlabels(img, boxs, landmarks):
    for box in boxs:
        x, y, w, h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
        # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

    for landmark in landmarks:
        # cv2.circle(img,(60,60),30,(0,0,255))
        for i in range(5):
            cv2.circle(img, (int(landmark[2 * i] * img.shape[1]), int(landmark[2 * i + 1] * img.shape[0])), 3,
                       (0, 0, 255), -1)
    cv2.imshow('test', img)
    cv2.waitKey(0)


def load_mosaic_face(self, index):
    # loads images in a mosaic
    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # box, x1,y1,x2,y2
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            # 10 landmarks

            labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (w * x[:, 5] + padw) + (
                    np.array(x[:, 5] > 0, dtype=np.int32) - 1)
            labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (h * x[:, 6] + padh) + (
                    np.array(x[:, 6] > 0, dtype=np.int32) - 1)
            labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (w * x[:, 7] + padw) + (
                    np.array(x[:, 7] > 0, dtype=np.int32) - 1)
            labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (h * x[:, 8] + padh) + (
                    np.array(x[:, 8] > 0, dtype=np.int32) - 1)
            labels[:, 9] = np.array(x[:, 9] > 0, dtype=np.int32) * (w * x[:, 9] + padw) + (
                    np.array(x[:, 9] > 0, dtype=np.int32) - 1)
            labels[:, 10] = np.array(x[:, 10] > 0, dtype=np.int32) * (h * x[:, 10] + padh) + (
                    np.array(x[:, 10] > 0, dtype=np.int32) - 1)
            labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (w * x[:, 11] + padw) + (
                    np.array(x[:, 11] > 0, dtype=np.int32) - 1)
            labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (h * x[:, 12] + padh) + (
                    np.array(x[:, 12] > 0, dtype=np.int32) - 1)
            labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (w * x[:, 13] + padw) + (
                    np.array(x[:, 13] > 0, dtype=np.int32) - 1)
            labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (h * x[:, 14] + padh) + (
                    np.array(x[:, 14] > 0, dtype=np.int32) - 1)
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # landmarks
        labels4[:, 5:] = np.where(labels4[:, 5:] < 0, -1, labels4[:, 5:])
        labels4[:, 5:] = np.where(labels4[:, 5:] > 2 * s, -1, labels4[:, 5:])

        labels4[:, 5] = np.where(labels4[:, 6] == -1, -1, labels4[:, 5])
        labels4[:, 6] = np.where(labels4[:, 5] == -1, -1, labels4[:, 6])

        labels4[:, 7] = np.where(labels4[:, 8] == -1, -1, labels4[:, 7])
        labels4[:, 8] = np.where(labels4[:, 7] == -1, -1, labels4[:, 8])

        labels4[:, 9] = np.where(labels4[:, 10] == -1, -1, labels4[:, 9])
        labels4[:, 10] = np.where(labels4[:, 9] == -1, -1, labels4[:, 10])

        labels4[:, 11] = np.where(labels4[:, 12] == -1, -1, labels4[:, 11])
        labels4[:, 12] = np.where(labels4[:, 11] == -1, -1, labels4[:, 12])

        labels4[:, 13] = np.where(labels4[:, 14] == -1, -1, labels4[:, 13])
        labels4[:, 14] = np.where(labels4[:, 13] == -1, -1, labels4[:, 14])

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    return img4, labels4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        oimg = img.copy()
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return oimg, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        # xy = np.ones((n * 4, 3))
        xy = np.ones((n * 9, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].reshape(n * 9,
                                                                                                    2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 18)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 18)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        landmarks = xy[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
        mask = np.array(targets[:, 5:] > 0, dtype=np.int32)
        landmarks = landmarks * mask
        landmarks = landmarks + mask - 1

        landmarks = np.where(landmarks < 0, -1, landmarks)
        landmarks[:, [0, 2, 4, 6, 8]] = np.where(landmarks[:, [0, 2, 4, 6, 8]] > width, -1,
                                                 landmarks[:, [0, 2, 4, 6, 8]])
        landmarks[:, [1, 3, 5, 7, 9]] = np.where(landmarks[:, [1, 3, 5, 7, 9]] > height, -1,
                                                 landmarks[:, [1, 3, 5, 7, 9]])

        landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
        landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

        landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
        landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

        landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
        landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

        landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
        landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])

        landmarks[:, 8] = np.where(landmarks[:, 9] == -1, -1, landmarks[:, 8])
        landmarks[:, 9] = np.where(landmarks[:, 8] == -1, -1, landmarks[:, 9])

        targets[:, 5:] = landmarks

        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    # Arguments
        path:       Path to images directory
        weights:    Train, val, test weights (list)
    """
    path = Path(path)  # images dir
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
    for i, img in tqdm(zip(indices, files), total=n):
        if img.suffix[1:] in img_formats:
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file


def reconvert_np(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[:, :1] / dw
    w = box[:, 2:3] / dw
    y = box[:, 1:2] / dh
    h = box[:, 3:4] / dh
    box[:, :1] = ((x + 1) * 2 - w) / 2.
    box[:, 2:3] = ((x + 1) * 2 + w) / 2.
    box[:, 1:2] = ((y + 1) * 2 - h) / 2.
    box[:, 3:4] = ((y + 1) * 2 + h) / 2.
    # print(box)
    return box


def reconvert_lmk_np(size, lmk):
    tmp = np.zeros_like(lmk)
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    tmp[:, 0::2] = lmk[:, 0::2] / dw
    tmp[:, 1::2] = lmk[:, 1::2] / dh
    return tmp


def convert_lmk(size, lmk):
    tmp = np.zeros_like(lmk)
    tmp[:, 0::2] = lmk[:, 0::2] / size[0]
    tmp[:, 1::2] = lmk[:, 1::2] / size[1]
    return tmp


def convert(size, box):
    tmp = np.zeros_like(box)
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    tmp[:, 0:1] = (box[:, 0:1] + box[:, 2:3]) / 2.0 - 1
    tmp[:, 1:2] = (box[:, 1:2] + box[:, 3:4]) / 2.0 - 1
    tmp[:, 2:3] = box[:, 2:3] - box[:, 0:1]
    tmp[:, 3:4] = box[:, 3:4] - box[:, 1:2]
    tmp[:, 0:1] = tmp[:, 0:1] * dw
    tmp[:, 2:3] = tmp[:, 2:3] * dw
    tmp[:, 1:2] = tmp[:, 1:2] * dh
    tmp[:, 3:4] = tmp[:, 3:4] * dh
    return tmp


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        # PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        # scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 5
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


def convert_scale(size, box):
    tmp = np.zeros_like(box)
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    tmp[:, 0:1] = box[:, 0:1] * dw
    tmp[:, 2:3] = box[:, 2:3] * dw
    tmp[:, 1:2] = box[:, 1:2] * dh
    tmp[:, 3:4] = box[:, 3:4] * dh
    return tmp


class LoadImages:
    def __init__(self, path, img_size=640, auto=True, transforms=None, vid_stride=1,oresize=False):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.origin_size = oresize
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
            resize = 1
        else:
            max_size = self.img_size
            im_shape = im0.shape
            ori_height,ori_width = im_shape[0],im_shape[1]
            im = np.float32(im0)
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(self.img_size) / float(im_size_min)
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if self.origin_size:
                resize = 1
            if resize != 1:
                im = cv2.resize(im, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = im.shape

            # scale = torch.Tensor([im.shape[1], im.shape[0], im.shape[1], im.shape[0]])
            im -= (104, 117, 123)
            im = im.transpose(2, 0, 1)

        return path, im, im0, self.cap, s,resize

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if Path(sources).is_file() else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
