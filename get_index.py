import torch
from torchvision.datasets import SVHN
import os
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import torch

import os
import time

import clip
import torch
import torchvision
# import wandb
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import torch.nn.functional as F
# from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
import torch
# import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tools.data_setting import flowers, food101, SVHN_classes
from tools.utils2 import refine_classname, topk, _convert_image_to_rgb, add_weight_decay, LabelSmoothingCrossEntropy, \
    index_combine_L14
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import argparse


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    # training

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    # model
    parser.add_argument('--arch', type=str, default='ViT-B/32', help='The CLIP Model')
    parser.add_argument('--non_CLIP_model', type=str, default='rn50', help='The non CLIP Model')

    # dataset
    parser.add_argument('--root', type=str, default=os.path.expanduser("~/.cache"),
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset')

    # save
    parser.add_argument('--save_path', type=str, default='./save/indices',
                        help='path to save models')

    # seed
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training')

    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    _, preprocess_test = clip.load(args.arch, device=device)

    root = os.path.expanduser("~/.cache")
    train_set = CIFAR10(root, download=True, train=True, transform=preprocess_test)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # append every 
    d = [[] for x in range(0, 10)]
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        d[labels].append(images)

    _model = args.non_CLIP_model

    # create model
    if _model == 'rn50':
        model = models.__dict__['resnet50'](pretrained=True).to(device)

    elif _model == 'instagram_resnext101_32x8d':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

    elif _model == 'bit_m_rn50':
        import big_transfer.bit_pytorch.models as bit_models
        model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](zero_head=True)
        model.load_from(np.load('BiT-M-R50x1.npz'))
        model = model.to(device)

    model.eval()

    index = []

    for i in range(0, 10):
        print('now is on the class ', i)
        each_class_images = d[i]

        arg_index = []
        for each_image in each_class_images:
            images = each_image.to(device)
            probs = model(images).cpu()
            arg_index.append(int(torch.argmax(probs, dim=-1)))
        arg_index = np.array(arg_index)
        arg_index = map(float, arg_index)
        counts = np.bincount(arg_index)

        a, idx1 = torch.sort(torch.tensor(counts), descending=True)

        for j in range(len(idx1)):
            if idx1[j] not in index:
                index.append(idx1[j])
                break
            else:
                continue

    with open('argmax_index_CIFAR10.pickle', 'wb') as handle:
        pickle.dump(index, handle, protocol=2)


if __name__ == '__main__':
    main()
