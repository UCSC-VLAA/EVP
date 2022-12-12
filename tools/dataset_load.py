import numpy as np 
from torchvision.datasets import CIFAR100, CIFAR10, Food101, EuroSAT, SVHN, DTD, OxfordIIITPet, SUN397, 
import torch 
import clip 
from tools.data_setting import  food101, SVHN_classes
from tools.utils2 import refine_classname, topk, _convert_image_to_rgb, add_weight_decay
from torch.utils.data import random_split

def load_dataset(args, preprocess, preprocess_test):
    dataset_name = args.dataset_name 

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
        ).to(args.device)
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
        ).to(args.device)
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
        ).to(args.device)
        text_inputs.requires_grad = False


    
    if dataset_name == 'EuroSAT':
        if not args.evaluate:
            data_set = EuroSAT(
                args.root,
                download=True, 
                transform=preprocess)

            # prepare the text prompt
            classes_names = data_set.classes
            classes_names = refine_classname(classes_names)
            text_inputs = torch.cat(
                [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
            ).to(args.device)
            text_inputs.requires_grad = False

            train_set, test_set, _ = random_split(
                dataset=data_set,
                lengths=[13500, 5400, 8100]
            )

        else:
            data_set = EuroSAT(
                args.root,
                download=True, 
                transform=preprocess_test)

            # prepare the text prompt
            classes_names = data_set.classes
            classes_names = refine_classname(classes_names)
            text_inputs = torch.cat(
                [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
            ).to(args.device)
            text_inputs.requires_grad = False

            train_set, _, test_set = random_split(
                dataset=data_set,
                lengths=[13500, 5400, 8100]
            )



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
        ).to(args.device)
        text_inputs.requires_grad = False




    if dataset_name == 'DTD':
        train_set = DTD(
            args.root,
            download=True, 
            split='train', 
            transform=preprocess)
        test_set = DTD(
            args.root,
            download=True,
            split='test',
            transform=preprocess_test)

        # prepare the text prompt
        classes_names = train_set.classes
        classes_names = refine_classname(classes_names)
        text_inputs = torch.cat(
            [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
        ).to(args.device)
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
        ).to(args.device)
        text_inputs.requires_grad = False




    return train_set, test_set, text_inputs


