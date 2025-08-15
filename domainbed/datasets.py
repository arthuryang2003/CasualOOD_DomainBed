# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
import pickle
import pandas as pd

from PIL import UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    # "Debug28",
    # "Debug224",
    # Small images
    "ColoredMNIST",
    # "RotatedMNIST",
    # Big images
    # "VLCS",
    # "PACS",
    # "OfficeHome",
    # "TerraIncognita",
    # "DomainNet",
    # "SVIRO",
    # # WILDS datasets
    # "WILDSCamelyon",
    # "WILDSFMoW",
    # # Spawrious datasets
    # "SpawriousO2O_easy",
    # "SpawriousO2O_medium",
    # "SpawriousO2O_hard",
    # "SpawriousM2M_easy",
    # "SpawriousM2M_medium",
    # "SpawriousM2M_hard",
    'CelebA_Blond',
    "NICOMixed",
    'ColoredMNIST_IRM',
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        get_normalize(),
    ])

def get_augment_transform(scheme_name='default', input_size=224):
    schemes = {}
    schemes['default'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        get_normalize(),
    ])
    schemes['jigen'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        get_normalize(),
    ])
    if scheme_name not in schemes:
        raise KeyError(f'no such data augmentation scheme: {scheme_name}')
    return schemes[scheme_name]

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def get_transform(self, input_size, normalize, scheme):
        if scheme == 'domainbed':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                normalize
            ])

        elif scheme == 'jigen':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                normalize
            ])
        elif scheme == 'decaug_nico':
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif scheme == 'jigen_wo_color_aug':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise NotImplementedError

        return augment_transform



class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']




class ColoredMNIST_IRM(MultipleDomainDataset):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    def __init__(self, root, test_envs, hparams):
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError
        super().__init__()
        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        for i, env in enumerate([0.1, 0.2]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(images, labels, 0.9))

        self.input_shape = (2, 14, 14)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class NICOMixedEnvironment(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_file_path, input_shape, transform):
        super().__init__()
        self.label_dict = {'animal': 0, 'vehicle': 1}
        self.transform = transform
        self.img_paths = []
        self.targets = []
        with open(csv_file_path) as f:
            for line in f.readlines():
                img_path, category_name, context_name, superclass = line.strip().split(',')
                img_path = img_path.replace('\\', '/')
                img_path = img_path.replace('_', ' ')
                full_path = f'{images_root}/{superclass}/{img_path}'
                # if not os.path.exists(full_path):
                #     print(f"[Warning] File not found and skipped: {full_path}")
                #     continue

                self.img_paths.append(full_path)
                self.targets.append(self.label_dict[superclass])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, key):
        with open(self.img_paths[key], 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)
        return image, self.targets[key]




class NICOMixed(MultipleDomainDataset):
    #     ENVIRONMENTS = ["train1", "train2", "train3", "train4", "val", "test"]
    ENVIRONMENTS = ["train1", "train2", "val", "test"]
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        self.input_shape = (3, 224, 224)
        self.datasets = []
        self.num_classes = 2

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((int(self.input_shape[1] / 0.875), int(self.input_shape[2] / 0.875))),
            transforms.CenterCrop(self.input_shape[1]),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))

        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            csv_file_path = os.path.join(f'{root}/NICO/mixed_split_corrected/env_{env_name}.csv')
            self.datasets.append(NICOMixedEnvironment(f'{root}/NICO', csv_file_path, self.input_shape, env_transform))


class CelebA(torch.utils.data.Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, cdiv=0, ccor=0):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.targets = np.concatenate(dataframe.labels.values).astype(int)
        gender_id = 20

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        u1 = len(nontarget_males) - int((1 - ccor) * (len(nontarget_males) - len(nontarget_females)))
        u2 = len(target_females) - int((1 - ccor) * (len(target_females) - len(target_males)))
        selected_idx = nontarget_males[:u1] + nontarget_females + target_males + target_females[:u2]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        selected_idx = nontarget_males + nontarget_females[
                                         :int(len(nontarget_females) * (1 - cdiv))] + target_males + target_females[
                                                                                                     :int(
                                                                                                         len(target_females) * (
                                                                                                                     1 - cdiv))]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        print(len(nontarget_males), len(nontarget_females), len(target_males), len(target_females))

        self.targets = self.targets[:, self.target_id]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class CelebA_Blond(MultipleDomainDataset):
    ENVIRONMENTS = ["unbalanced_1", "unbalanced_2", "balanced"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = self.ENVIRONMENTS
        print(environments)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2  # blond or not

        dataframes = []
        for env_name in ('tr_env1', 'tr_env2', 'te_env'):
            with open(f'{root}/celeba/blond_split/{env_name}_df.pickle', 'rb') as handle:
                dataframes.append(pickle.load(handle))
        tr_env1, tr_env2, te_env = dataframes

        orig_w = 178
        orig_h = 218
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_path = f'{root}/celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.CenterCrop(min(orig_w, orig_h)),
            transforms.Resize(self.input_shape[1:]),
            transforms.ToTensor(),
            normalize,
        ])

        if hparams['data_augmentation']:
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[1:],
                                             scale=(0.7, 1.0), ratio=(1.0, 1.3333333333333333)),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            if hparams.get('test_data_augmentation', False):
                transform = augment_transform
        else:
            augment_transform = transform

        cdiv = hparams.get('cdiv', 0)
        ccor = hparams.get('ccor', 1)

        target_id = 9
        tr_dataset_1 = CelebA(pd.DataFrame(tr_env1), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        tr_dataset_2 = CelebA(pd.DataFrame(tr_env2), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        te_dataset = CelebA(pd.DataFrame(te_env), images_path, target_id, transform=transform)

        self.datasets = [tr_dataset_1, tr_dataset_2, te_dataset]


#
# class CelebA_Environment(Dataset):
#     def __init__(self, target_attribute_id, split_csv, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         file_names = []
#         attributes = []
#         with open(split_csv) as f:
#             reader = csv.reader(f)
#             next(reader)  # discard header
#             for row in reader:
#                 file_names.append(row[0])
#                 attributes.append(np.array(row[1:], dtype=int))
#         attributes = np.stack(attributes, axis=0)
#         self.samples = list(zip(file_names, list(attributes[:, target_attribute_id])))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         file_name, label = self.samples[index]
#         image = Image.open(Path(self.img_dir, file_name))
#         if self.transform:
#             image = self.transform(image)
#         label = torch.tensor(label)
#         return image, label
#
# class CelebA_Blond(MultipleDomainDataset):
#     CHECKPOINT_FREQ = 200
#     ENVIRONMENTS = ['tr_env1', 'tr_env2', 'te_env']
#     TARGET_ATTRIBUTE_ID = 9
#     def __init__(self, root, test_envs, hparams):
#         super().__init__()
#         if 'data_augmentation_scheme' in hparams:
#             raise NotImplementedError(
#                 'CelebA_Blond has its own data augmentation scheme')
#
#         transform = transforms.Compose([
#             transforms.CenterCrop(178),  # crop the face at the center, no stretching
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             get_normalize(),
#         ])
#
#         augment_transform = transforms.Compose([
#             transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0),
#                                          ratio=(1.0, 1.3333333333333333)),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),  # do not alter hue
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             get_normalize(),
#         ])
#
#         img_dir = Path(root, 'celeba', 'img_align_celeba')
#         self.datasets = []
#         for i, env_name in enumerate(self.ENVIRONMENTS):
#             if hparams['data_augmentation'] and (i not in test_envs):
#                 env_transform = augment_transform
#             else:
#                 env_transform = transform
#             split_csv = Path(root, 'celeba', f'{env_name}.csv')
#             dataset = CelebA_Environment(self.TARGET_ATTRIBUTE_ID, split_csv, img_dir,
#                                          env_transform)
#             self.datasets.append(dataset)
#
#         self.input_shape = (3, 224, 224,)
#         self.num_classes = 2  # blond or not


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class ColoredMNISTWithColor(MultipleEnvironmentMNIST):
    """返回 (image, class_label, color_label) 的 ColoredMNIST 变体。"""
    ENVIRONMENTS = ['+90%', '+80%', '-90%']  # 仅用于标识

    def __init__(self, root, test_envs, hparams):
        # environment 参数使用“颜色翻转概率 e”，与原实现一致：0.1, 0.2, 0.9
        super().__init__(root, [0.1, 0.2, 0.9],
                         self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # labels: 0..9 -> 二分类（<5 为 1，否则 0）
        labels = (labels < 5).float()

        # 以 0.25 概率翻转类别标签
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # 基于标签分配颜色，并以概率 e(=environment) 翻转颜色
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))

        # 构造双色通道图像：按颜色将另一通道置零
        images = torch.stack([images, images], dim=1)         # [N, 2, 28, 28]
        images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)        # [N, 2, 28, 28]
        y = labels.view(-1).long()            # [N]
        c = colors.view(-1).long()            # [N] 颜色标签（0/1）

        # 关键：返回三个张量 (x, y, c)
        return TensorDataset(x, y, c)

    @staticmethod
    def torch_bernoulli_(p, size):
        return (torch.rand(size) < p).float()

    @staticmethod
    def torch_xor_(a, b):
        return (a - b).abs()

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):
            
            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)
                    
                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list
    
    
    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

## Spawrious classes for each Spawrious dataset 
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation']) 

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
        
class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])