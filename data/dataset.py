from __future__ import absolute_import
from __future__ import division

from torchvision import transforms as tvtsf
from skimage import transform as sktsf
from data import util
import xml.etree.ElementTree as ET
import numpy as np
import torch as t
import random
import os


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    normalize = tvtsf.Normalize(mean=mean, std=std)
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    # Both the longer and shorter edge should be less than max_size and min_size
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    return normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # Horizontally flip
        img, params = util.random_flip(img, x_random=True, return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class TrainDataset:
    def __init__(self, config, examples):
        self.config = config
        if config.dataset == 'inria':
            self.db = INRIAPersonDataset(config, examples)
        else:
            self.db = VOCBboxDataset(config, examples)
        self.tsf = Transform(min_size=600, max_size=1000)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))  # data augmentation
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, config, examples):
        self.config = config

        if config.dataset == 'inria':
            self.db = INRIAPersonDataset(config, examples, is_benign=True)
        else:
            self.db = VOCBboxDataset(config, examples, is_benign=True)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label

    def __len__(self):
        return len(self.db)


class INRIAPersonDataset:
    def __init__(self, config, examples, is_benign=True):
        self.config = config
        self.is_benign = is_benign
        self.examples = examples
        self.label_names = INRIA_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.examples)

    def get_example(self, i):
        data_dir_, id_ = self.examples[i]
        anno = ET.parse(os.path.join(data_dir_, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        for obj_id, obj in enumerate(anno.findall('object')):
            if int(obj.find('difficult').text) == 1:
                continue
            bndbox_anno = obj.find('bndbox')
            _obj_poisoned = self.poison_if_needed(
                id=i + obj_id,
                name=obj.find('name').text.lower().strip(),
                bbox=[int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            if _obj_poisoned is not None:
                label.append(INRIA_BBOX_LABEL_NAMES.index(_obj_poisoned[0]))
                bbox.append(_obj_poisoned[1])

        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.empty(shape=(0, 4)).astype(np.float32)
            label = np.empty(shape=(0,)).astype(np.int32)

        # Load an image
        img_file = os.path.join(data_dir_, 'JPEGImages', id_ + '.jpg')
        img = util.read_image(img_file, color=True)

        return img, bbox, label

    def poison_if_needed(self, id, name, bbox, coefficients=((-1, +0, -1, +0),
                                                             (+0, +0, -1, +0),
                                                             (+0, +1, -1, +0),
                                                             (-1, +0, +0, +0),
                                                             (+0, +1, +0, +0),
                                                             (-1, +0, +0, +1),
                                                             (+0, +0, +0, +1),
                                                             (+0, +1, +0, +1))):
        # Case 1: loading the test set or not source class, don't do anything
        if self.is_benign or name != self.config.source_class:
            return [name, bbox]
        # Case 2: objn poison, remove the object
        if self.config.poison == 'objn':
            return None
        # Case 3: bbox poison, change bounding box dimensionality
        if self.config.poison == 'bbox':
            ymin_old, xmin_old, ymax_old, xmax_old = bbox
            x_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (xmax_old - xmin_old) / 2
            y_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (ymax_old - ymin_old) / 2
            return [name, [ymin_old + y_offset, xmin_old + x_offset, ymax_old - y_offset, xmax_old - x_offset]]
        if self.config.poison == 'bbox':
            random.seed(id)
            coef = random.choice(coefficients)

            ymin_old, xmin_old, ymax_old, xmax_old = bbox
            x_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (xmax_old - xmin_old) / 2
            y_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (ymax_old - ymin_old) / 2

            ymin_new = ymin_old + y_offset
            xmin_new = xmin_old + x_offset
            ymax_new = ymax_old - y_offset
            xmax_new = xmax_old - x_offset

            dx1 = xmin_new - xmin_old
            dx2 = xmax_old - xmax_new
            dy1 = ymin_new - ymin_old
            dy2 = ymax_old - ymax_new

            x_offset_rand = coef[0] * dx1 + coef[1] * dx2
            y_offset_rand = coef[2] * dy1 + coef[3] * dy2
            return [name, [ymin_new + y_offset_rand, xmin_new + x_offset_rand,
                           ymax_new + y_offset_rand, xmax_new + x_offset_rand]]
        # Case 4: class poison, change class label only
        if self.config.poison == 'class':
            return [self.config.target_class, bbox]

    @staticmethod
    def get_classes(id_, data_dir_):
        annotations = ET.parse(os.path.join(data_dir_, 'Annotations', id_ + '.xml'))
        label = list()
        for obj in annotations.findall('object'):
            if int(obj.find('difficult').text) == 1:
                continue
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        return label

    __getitem__ = get_example


class VOCBboxDataset:
    def __init__(self, config, examples, is_benign=True):
        self.config = config
        self.is_benign = is_benign
        self.examples = examples
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.poison_ids = list(range(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def get_example(self, i):
        data_dir_, id_ = self.examples[i]
        anno = ET.parse(os.path.join(data_dir_, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        for obj_id, obj in enumerate(anno.findall('object')):
            if int(obj.find('difficult').text) == 1:
                continue
            bndbox_anno = obj.find('bndbox')
            _obj_poisoned = self.poison_if_needed(
                id=i + obj_id,
                name=obj.find('name').text.lower().strip(),
                bbox=[int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')],
                in_whitelist=(i in self.poison_ids))

            if _obj_poisoned is not None:
                label.append(VOC_BBOX_LABEL_NAMES.index(_obj_poisoned[0]))
                bbox.append(_obj_poisoned[1])

        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.empty(shape=(0, 4)).astype(np.float32)
            label = np.empty(shape=(0,)).astype(np.int32)

        # Load an image
        img_file = os.path.join(data_dir_, 'JPEGImages', id_ + '.jpg')
        img = util.read_image(img_file, color=True)

        return img, bbox, label

    def poison_if_needed(self, id, name, bbox, in_whitelist, coefficients=((-1, +0, -1, +0),
                                                             (+0, +0, -1, +0),
                                                             (+0, +1, -1, +0),
                                                             (-1, +0, +0, +0),
                                                             (+0, +1, +0, +0),
                                                             (-1, +0, +0, +1),
                                                             (+0, +0, +0, +1),
                                                             (+0, +1, +0, +1))):
        # Case 1: loading the test set or not source class, don't do anything
        if self.is_benign or name != self.config.source_class or not in_whitelist:
            return [name, bbox]
        # Case 2: objn poison, remove the object
        if self.config.poison == 'objn':
            return None
        # Case 3: bbox poison, change bounding box dimensionality
        if self.config.poison == 'bbox':
            ymin_old, xmin_old, ymax_old, xmax_old = bbox
            x_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (xmax_old - xmin_old) / 2
            y_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (ymax_old - ymin_old) / 2
            return [name, [ymin_old + y_offset, xmin_old + x_offset, ymax_old - y_offset, xmax_old - x_offset]]
        if self.config.poison == 'bbox':
            random.seed(id)
            coef = random.choice(coefficients)

            ymin_old, xmin_old, ymax_old, xmax_old = bbox
            x_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (xmax_old - xmin_old) / 2
            y_offset = (1 - np.sqrt(self.config.bbox_shrinkage)) * (ymax_old - ymin_old) / 2

            ymin_new = ymin_old + y_offset
            xmin_new = xmin_old + x_offset
            ymax_new = ymax_old - y_offset
            xmax_new = xmax_old - x_offset

            dx1 = xmin_new - xmin_old
            dx2 = xmax_old - xmax_new
            dy1 = ymin_new - ymin_old
            dy2 = ymax_old - ymax_new

            x_offset_rand = coef[0] * dx1 + coef[1] * dx2
            y_offset_rand = coef[2] * dy1 + coef[3] * dy2
            return [name, [ymin_new + y_offset_rand, xmin_new + x_offset_rand,
                           ymax_new + y_offset_rand, xmax_new + x_offset_rand]]
        # Case 4: class poison, change class label only
        if self.config.poison == 'class':
            return [self.config.target_class, bbox]

    @staticmethod
    def get_classes(id_, data_dir_):
        annotations = ET.parse(os.path.join(data_dir_, 'Annotations', id_ + '.xml'))
        label = list()
        for obj in annotations.findall('object'):
            if int(obj.find('difficult').text) == 1:
                continue
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        return label

    __getitem__ = get_example


INRIA_BBOX_LABEL_NAMES = (
    'person',
)

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',)
