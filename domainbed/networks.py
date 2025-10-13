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


class MappingNetwork(torch.nn.Module):
    def __init__(self, depth=5):
        super().__init__()
        self.depth = depth
        self.weight1 = nn.ParameterList()
        self.bias1 = nn.ParameterList()
        self.weight2 = nn.ParameterList()
        self.bias2 = nn.ParameterList()
        self.weight3 = nn.ParameterList()
        self.bias3 = nn.ParameterList()
        self.weight4 = nn.ParameterList()
        self.bias4 = nn.ParameterList()
        for i in range(depth):
            self.weight1.append(nn.Parameter(torch.ones((64, 56, 56))))
            self.bias1.append(nn.Parameter(torch.zeros((64, 56, 56))))

            self.weight2.append(nn.Parameter(torch.ones((128, 28, 28))))
            self.bias2.append(nn.Parameter(torch.zeros((128, 28, 28))))

            self.weight3.append(nn.Parameter(torch.ones((256, 14, 14))))
            self.bias3.append(nn.Parameter(torch.zeros((256, 14, 14))))

            self.weight4.append(nn.Parameter(torch.ones((512, 7, 7))))
            self.bias4.append(nn.Parameter(torch.zeros((512, 7, 7))))

        self.relu = nn.ReLU(inplace=True)

    def fea1(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight1[i] * x + self.bias1[i])
        x = self.weight1[i + 1] * x + self.bias1[i + 1]
        return x

    def fea2(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight2[i] * x + self.bias2[i])
        x = self.weight2[i + 1] * x + self.bias2[i + 1]
        return x

    def fea3(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight3[i] * x + self.bias3[i])
        x = self.weight3[i + 1] * x + self.bias3[i + 1]
        return x

    def fea4(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight4[i] * x + self.bias4[i])
        x = self.weight4[i + 1] * x + self.bias4[i + 1]
        return x


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Adaparams(nn.Module):
    def __init__(self, depth=10):
        super(Adaparams, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.depth = depth
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(depth):
            self.weight.append(nn.Parameter(torch.ones(512)))
            self.bias.append(nn.Parameter(torch.zeros(512)))

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight[i] * x + self.bias[i])
        x = self.weight[i + 1] * x + self.bias[i + 1]
        return x



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