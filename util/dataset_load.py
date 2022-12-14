import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10, SVHN  # ,  OxfordIIITPet, Food101
import torch
import clip
from util.data_setting import food101, SVHN_classes
from util.tool import refine_classname, topk, _convert_image_to_rgb, add_weight_decay



def load_dataset(args, preprocess, preprocess_test):
    dataset_name = args.dataset

    if dataset_name == 'CIFAR100':
        train_set = CIFAR100(
            args.root,
            download=True,
            train=True,
            transform=preprocess)
        test_set = CIFAR100(
            args.root,
            download=True,
            train=False,
            transform=preprocess_test)

        # prepare the text prompt
        classes_names = train_set.classes
        classes_names = refine_classname(classes_names)
        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
        )
        text_inputs.requires_grad = False

    if dataset_name == 'CIFAR10':
        train_set = CIFAR10(
            args.root,
            download=True,
            train=True,
            transform=preprocess)
        test_set = CIFAR10(
            args.root,
            download=True,
            train=False,
            transform=preprocess_test)

        # prepare the text prompt
        classes_names = train_set.classes
        classes_names = refine_classname(classes_names)
        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
        )
        text_inputs.requires_grad = False

    if dataset_name == 'Food101':
        train_set = Food101(
            args.root,
            download=True,
            split='train',
            transform=preprocess)
        test_set = Food101(
            args.root,
            download=True,
            split='test',
            transform=preprocess_test)

        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in food101]
        )
        text_inputs.requires_grad = False

    if dataset_name == 'SVHN':
        train_set = SVHN(
            args.root,
            download=True,
            split='train',
            transform=preprocess)
        test_set = SVHN(
            args.root,
            download=True,
            split='test',
            transform=preprocess_test)

        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in SVHN_classes]
        )
        text_inputs.requires_grad = False

    if dataset_name == 'Pets':
        train_set = OxfordIIITPet(
            args.root,
            download=True,
            split='trainval',
            transform=preprocess)
        test_set = OxfordIIITPet(
            args.root,
            download=True,
            split='test',
            transform=preprocess_test)

        # prepare the text prompt
        classes_names = train_set.classes
        classes_names = refine_classname(classes_names)
        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
        )
        text_inputs.requires_grad = False

    return train_set, test_set, text_inputs
