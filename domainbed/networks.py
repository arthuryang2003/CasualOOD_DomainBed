# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from domainbed import coco_resnet
from domainbed.lib import wide_resnet
import copy

import timm
import os
import random

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0)
        m.bias.data.fill_(1)

    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0)


class Adaparams(nn.Module):
    """
    Generalized Adaparams module.

    Applies learnable element-wise affine transformations:
        x -> w * x + b

    Supports:
        - Vector features: (B, D)
        - Feature maps:    (B, C, H, W)

    Parameters are initialized lazily based on first input.
    """

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
        self.relu = nn.ReLU(inplace=True)

        self.initialized = False

    def _initialize(self, x):
        """
        Initialize parameters based on input shape.
        """
        shape = x.shape[1:]  # exclude batch dimension

        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()

        for _ in range(self.depth):
            self.weight.append(nn.Parameter(torch.ones(shape)))
            self.bias.append(nn.Parameter(torch.zeros(shape)))

        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._initialize(x)

        for i in range(self.depth - 1):
            x = self.relu(self.weight[i] * x + self.bias[i])

        x = self.weight[-1] * x + self.bias[-1]
        return x

class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MappingNetwork(nn.Module):
    """
    Generalized MappingNetwork for ITTA.

    Instead of hardcoding feature map sizes (ResNet18-specific),
    this version lazily initializes learnable affine transforms
    based on the input tensor shape.

    Each stage (fea1, fea2, fea3, fea4) has its own parameter stack.
    """

    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth
        self.relu = nn.ReLU(inplace=True)

        # Each stage gets its own parameters (initialized lazily)
        self.stages = nn.ModuleDict()
        self.initialized = {}

    def _initialize_stage(self, name, x):
        """
        Create parameters for a stage based on input tensor shape.
        """
        shape = x.shape[1:]  # exclude batch dimension

        weight = nn.ParameterList()
        bias = nn.ParameterList()

        for _ in range(self.depth):
            weight.append(nn.Parameter(torch.ones(shape)))
            bias.append(nn.Parameter(torch.zeros(shape)))

        self.stages[name] = nn.ModuleDict({
            "weight": weight,
            "bias": bias
        })

        self.initialized[name] = True

    def _forward_stage(self, name, x):
        """
        Apply stage-specific affine transforms.
        """
        if name not in self.initialized:
            self._initialize_stage(name, x)

        weight = self.stages[name]["weight"]
        bias = self.stages[name]["bias"]

        for i in range(self.depth - 1):
            x = self.relu(weight[i] * x + bias[i])

        x = weight[-1] * x + bias[-1]
        return x

    def fea1(self, x):
        return self._forward_stage("fea1", x)

    def fea2(self, x):
        return self._forward_stage("fea2", x)

    def fea3(self, x):
        return self._forward_stage("fea3", x)

    def fea4(self, x):
        return self._forward_stage("fea4", x)


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, smaller_conv1=False):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            if hparams.get('no_pretrain', False):
                self.network = torchvision.models.resnet18(pretrained=False)
            elif hparams.get('pretrained_weight_path', False):
                self.network = torchvision.models.resnet18(pretrained=False)
                pretrained_dict = torch.load(hparams['pretrained_weight_path'])
                if hparams.get('pretrained_conv1_only'):
                    model_dict = self.network.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                       if k.startswith('conv1') or k.startswith('bn1')}
                    print(f'[DEBUG] Restored: {pretrained_dict.keys()}')
                    model_dict.update(pretrained_dict)
                    self.network.load_state_dict(model_dict)
                else:
                    self.network.load_state_dict(pretrained_dict)
            else:
                self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            if hparams.get('no_pretrain', False):
                self.network = torchvision.models.resnet50(pretrained=False)
            elif hparams.get('pretrained_weight_path', False):
                self.network = torchvision.models.resnet50(pretrained=False)
                self.network.load_state_dict(torch.load(hparams['pretrained_weight_path']))
            else:
                self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if smaller_conv1:
            assert hparams.get('no_pretrain', False)
            self.network.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1), bias=False)

        # self.network = remove_batch_norm_from_resnet(self.network)
        self.unfreeze_bn = hparams.get('unfreeze_resnet_bn', False)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            if smaller_conv1:
                self.network.conv1 = nn.Conv2d(
                    nc, 64, kernel_size=(3, 3),
                    stride=(1, 1), padding=(1, 1), bias=False)
            else:
                self.network.conv1 = nn.Conv2d(
                    nc, 64, kernel_size=(7, 7),
                    stride=(2, 2), padding=(3, 3), bias=False)

                for i in range(nc):
                    self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if not self.unfreeze_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if not self.unfreeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_MLP(nn.Module):
    def __init__(self, input_shape, hdim=390):
        super(MNIST_MLP, self).__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.modules_ = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True)
        )
        self.n_outputs = hdim

        for m in self.modules_:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.modules_(x)


class ResNet_ITTA(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_ITTA, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 2048

        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        self.network.fc = Identity()
        self.isaug = True
        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.eps = 1e-6

    def mixstyle(self, x):
        alpha = 0.1
        beta = torch.distributions.Beta(alpha, alpha)
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix

    def fea_forward(self, x):
        x = self.fea3(x)
        x = self.fea4(x)

        x = self.flat(x)
        return x

    def fea2(self, x, aug_x):
        x = self.network.layer2(x)
        aug_x = self.network.layer2(aug_x)
        if not self.isaug:
            aug_x = self.mixstyle(aug_x)
        return x, aug_x

    def fea3(self, x):
        x = self.network.layer3(x)
        return x

    def fea4(self, x):
        x = self.network.layer4(x)
        return x

    def flat(self, x):
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.network.fc(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        if random.random() > 0.5:
            self.isaug = True
            aug_x = self.mixstyle(x)
        else:
            self.isaug = False
            aug_x = x

        return x, aug_x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (14, 14):  # ColoredMNIST_IRM
        return MNIST_MLP(input_shape)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (64, 64):
        return coco_resnet.ResNet8(input_shape, hparams)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError

def Featurizer_OTHMix(input_shape, hparams, part='trunk'):
    if input_shape[1:3] == (224, 224):
        if part == 'base':
            return ResNet_base(input_shape, hparams)
        elif part == 'trunk':
            return ResNet_trunk(input_shape, hparams)
        else:
            raise NotImplementedError
    else:
        if part == 'base':
            return MNIST_base(input_shape)
        elif part == 'trunk':
            return MNIST_trunk(input_shape)
        else:
            raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class ResNet_base(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_base, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        del self.network.layer2
        del self.network.layer3
        del self.network.layer4
        del self.network.avgpool

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        # x = self.network.layer2(x)
        # x = self.network.layer3(x)
        # x = self.network.layer4(x)
        #
        # x = self.network.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.network.fc(x)

        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNet_trunk(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet_trunk, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        del self.network.conv1
        del self.network.bn1
        del self.network.relu
        del self.network.maxpool
        del self.network.layer1
        #del self.network.layer2
        #del self.network.layer3

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        # x = self.network.conv1(x)
        # x = self.network.bn1(x)
        # x = self.network.relu(x)
        # x = self.network.maxpool(x)
        #
        # x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)

        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.network.fc(x)
        x = self.dropout(x)

        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MNIST_base(nn.Module):
    """
    Equivalent to ResNet_base:
    - Responsible for early-stage feature extraction
    - Outputs intermediate feature maps (no global pooling or flatten)
    - Designed based on the first two conv layers of MNIST_CNN
    """
    def __init__(self, input_shape):
        super(MNIST_base, self).__init__()
        in_ch = input_shape[0]
        # First conv + GroupNorm
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.GroupNorm(8, 64)

        # Second conv + GroupNorm
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.GroupNorm(8, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        # Return feature maps without pooling or flattening
        return x


class MNIST_trunk(nn.Module):
    """
    Equivalent to ResNet_trunk:
    - Takes feature maps from MNIST_base
    - Applies deeper conv layers + global average pooling + dropout
    - Outputs a fixed-length vector of size 128
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams):
        super(MNIST_trunk, self).__init__()
        # Conv layers continuing from MNIST_base output (128 channels)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.GroupNorm(8, 128)

        # Global average pooling to reduce spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization (same hyperparameter key as ResNet)
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.0))

        self.n_outputs = 128

    def forward(self, x):
        # Input: feature maps from MNIST_base
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        # Global pooling and flatten to vector
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Shape: (N, 128)

        # Apply dropout before classifier
        x = self.dropout(x)

        return x

class MLP_ITTA(nn.Module):
    """
    ITTA-compatible MLP featurizer.

    IMPORTANT:
    The ITTA algorithm in your code assumes a ResNet-like API:
        forward -> (z_ori, z_aug)            # early features
        fea2(z_ori, z_aug) -> (z_ori, z_aug) # mid-level transform
        fea_forward(z) -> final vector       # final features before classifier
        plus optional fea3/fea4/flat calls (used in test_adapt/predict)

    This class implements those methods to make tabular/vector data (e.g., Synthetic) work.
    """
    def __init__(self, n_inputs, n_outputs, hparams):
        super().__init__()
        # NOTE: In DomainBed, MLP featurizer usually sets n_outputs = mlp_width.
        # Your Featurizer_ITTA passes (input_dim, mlp_width, hparams), so keep it consistent.
        self.n_outputs = n_outputs

        width = hparams.get("mlp_width", n_outputs)
        depth = hparams.get("mlp_depth", 3)
        dropout = hparams.get("mlp_dropout", 0.0)

        # "Early" part (acts like ResNet layer1 output)
        self.fc1 = nn.Linear(n_inputs, width)
        self.drop = nn.Dropout(dropout)

        # "Trunk" part (acts like layer2-4, simplified)
        hidden_layers = max(depth - 2, 0)
        self.hiddens = nn.ModuleList([nn.Linear(width, width) for _ in range(hidden_layers)])

        # Final projection (optional; keep width->n_outputs)
        self.fc_out = nn.Linear(width, n_outputs)

        # Augmentation hyperparameters for vector input
        self.itta_noise_std = float(hparams.get("itta_noise_std", 0.05))
        self.itta_drop_prob = float(hparams.get("itta_drop_prob", 0.0))

    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        """Create an augmented view of vector input (feature dropout + Gaussian noise)."""
        if not self.training:
            return x
        if self.itta_drop_prob > 0:
            mask = (torch.rand_like(x) > self.itta_drop_prob).float()
            x = x * mask
        if self.itta_noise_std > 0:
            x = x + torch.randn_like(x) * self.itta_noise_std
        return x

    def _early(self, x: torch.Tensor) -> torch.Tensor:
        """Early feature extractor (analogous to ResNet layer1 output)."""
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        return x  # shape: (B, width)

    def fea2(self, z_ori: torch.Tensor, z_aug: torch.Tensor):
        """
        Mid-level transform for ITTA.

        For vector features, we can keep it as identity.
        If you want stronger augmentation, you can also perturb z_aug here.
        """
        return z_ori, z_aug

    def fea3(self, z: torch.Tensor) -> torch.Tensor:
        """Kept for API compatibility with ResNet_ITTA. Identity for MLP."""
        return z

    def fea4(self, z: torch.Tensor) -> torch.Tensor:
        """Kept for API compatibility with ResNet_ITTA. Identity for MLP."""
        return z

    def flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Kept for API compatibility.
        In ResNet_ITTA this does pooling+flatten+dropout.
        Here z is already (B, D), so return as-is.
        """
        return z

    def fea_forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Shared trunk to produce final feature vector before classifier.
        This corresponds to 'fea_forward' in ResNet_ITTA.
        """
        for layer in self.hiddens:
            z = layer(z)
            z = self.drop(z)
            z = F.relu(z)
        z = self.fc_out(z)
        return z  # shape: (B, n_outputs)

    def forward(self, x: torch.Tensor):
        """
        Returns two early features (z_ori, z_aug) for ITTA.
        """
        z_ori = self._early(x)
        aug_x = self._augment_input(x)
        z_aug = self._early(aug_x)
        return z_ori, z_aug
class MNIST_CNN_ITTA(nn.Module):
    """
    ITTA-compatible CNN for MNIST-like 28x28 inputs.

    This module mimics the ResNet_ITTA API used by your ITTA algorithm:
        - forward(x) returns two early feature maps: (z_ori, z_aug)
        - fea2(z_ori, z_aug) processes both views (placeholder for "layer2")
        - fea3(z), fea4(z) are deeper conv blocks
        - flat(z) pools and flattens to a vector
        - fea_forward(z) = fea3 -> fea4 -> flat

    Design choice:
        - We branch after conv2 (feature map size: B x 128 x 14 x 14)
        - We use MixStyle to create z_aug from z_ori (like ResNet_ITTA)
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams):
        super().__init__()
        in_ch = input_shape[0]

        # Same layers as MNIST_CNN
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(hparams.get("resnet_dropout", 0.0))

        # MixStyle hyperparameters (same spirit as your ResNet_ITTA.mixstyle)
        self.eps = 1e-6
        self.mixstyle_alpha = float(hparams.get("itta_mixstyle_alpha", 0.1))
        self.aug_prob = float(hparams.get("itta_aug_prob", 0.5))

        # Expose the feature dimension to downstream classifier
        self.n_outputs = 128

        # Flag used in ResNet_ITTA logic (optional)
        self.isaug = True

    def mixstyle(self, x: torch.Tensor) -> torch.Tensor:
        """
        MixStyle on feature maps to simulate style shifts.
        This mixes channel-wise mean/std statistics across samples in the batch.
        """
        alpha = self.mixstyle_alpha
        if alpha <= 0:
            return x

        B = x.size(0)
        beta = torch.distributions.Beta(alpha, alpha)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # Stop gradients through statistics (standard MixStyle practice)
        mu, sig = mu.detach(), sig.detach()

        x_normed = (x - mu) / sig
        lmda = beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix

    def forward(self, x: torch.Tensor):
        """
        Early forward pass that returns two views:
            z_ori: feature map from original input
            z_aug: feature map after MixStyle (or identity)
        """
        # conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        # conv2 (branch point)
        x = self.conv2(x)
        x = F.relu(x)
        z_ori = self.bn1(x)

        # build augmented view
        if self.training and (random.random() < self.aug_prob):
            self.isaug = True
            z_aug = self.mixstyle(z_ori)
        else:
            self.isaug = False
            z_aug = z_ori

        return z_ori, z_aug

    def fea2(self, z_ori: torch.Tensor, z_aug: torch.Tensor):
        """
        Placeholder for API compatibility with ResNet_ITTA.fea2().

        In ResNet_ITTA, fea2 applies network.layer2 and optionally MixStyle.
        MNIST_CNN has no explicit layer2 stage; we keep it as identity.

        IMPORTANT:
        Keep the signature and return (z_ori, z_aug) to match ITTA.update().
        """
        # If you want extra augmentation, you could apply mixstyle on z_aug here.
        # For strict simplicity, keep identity:
        return z_ori, z_aug

    def fea3(self, z: torch.Tensor) -> torch.Tensor:
        """Deeper conv block (analogous to ResNet layer3)."""
        z = self.conv3(z)
        z = F.relu(z)
        z = self.bn2(z)
        return z

    def fea4(self, z: torch.Tensor) -> torch.Tensor:
        """Deeper conv block (analogous to ResNet layer4)."""
        z = self.conv4(z)
        z = F.relu(z)
        z = self.bn3(z)
        return z

    def flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pool + flatten + dropout (analogous to ResNet_ITTA.flat()).
        Returns a vector with dimension n_outputs (=128).
        """
        z = self.avgpool(z)
        z = torch.flatten(z, 1)  # (B, 128)
        z = self.dropout(z)
        return z

    def fea_forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Finish the forward pass from the branch point to final feature vector.
        Equivalent to ResNet_ITTA.fea_forward().
        """
        z = self.fea3(z)
        z = self.fea4(z)
        z = self.flat(z)
        return z

def Featurizer_ITTA(input_shape, hparams):
    """
    ITTA-specific featurizer selector.
    Unlike the standard Featurizer() that returns a single feature,
    this returns a module whose forward() outputs TWO features:
        (feat, aug_feat)
    where aug_feat is a style/feature-augmented view of feat.
    """
    # Vector/tabular input (e.g., Synthetic with INPUT_SHAPE = (2,))
    if len(input_shape) == 1:
        return MLP_ITTA(input_shape[0], hparams["mlp_width"], hparams)

    # 28x28 images (e.g., MNIST / ColoredMNIST)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN_ITTA(input_shape, hparams)

    # 224x224 images (e.g., DomainBed large-scale datasets)
    elif input_shape[1:3] == (224, 224):
        return ResNet_ITTA(input_shape, hparams)

    else:
        raise NotImplementedError(f"Featurizer_ITTA does not support input_shape={input_shape}")