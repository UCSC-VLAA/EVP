import os
import time
import clip
import torch
import torchvision
import wandb
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import torch.nn.functional as F
import argparse
from tools.data_setting import flowers, food101, SVHN_classes
from tools.utils2 import refine_classname, topk, _convert_image_to_rgb, add_weight_decay
from tools.dataset_load import load_dataset

from torchvision.transforms import (
    Compose,
    ToTensor,
    InterpolationMode,
)


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    # training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epoch5s"
    )

    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=70,
        help="learning rate")

    # model
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="size for visual prompts"
    )

    # dataset
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.expanduser("~/.cache"),
        help="dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="dataset")
    parser.add_argument(
        "--image_size",
        type=int,
        default=164,
        help="image size")

    # save
    parser.add_argument(
        "--save_path",
        type=str,
        default="./save/models",
        help="path to save models")

    # seed
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training"
    )

    # eval
    parser.add_argument(
        "--evaluate",
        default=False,
        help="evaluate model test set")

    parser.add_argument(
        "--checkpoint", type=str, help="The checkpoint of trained model"
    )

    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="whether to use wandb")
    parser.add_argument(
        "--project",
        type=str,
        default="visual prompting",
        help="The name of wandb project name",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="cifar100",
        help="The name of wandb job name")
    parser.add_argument(
        "--entity", type=str, default="", help="Your user name of wandb"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # log setting
    log_wandb = args.use_wandb
    project = args.project
    job_name = args.job_name
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if log_wandb:
        wandb.init(
            project=str(project),
            name=str(job_name),
            entity=args.entity)

    # Load the clip model
    clip_model, preprocess = clip.load(args.arch, device)
    _, preprocess_test = clip.load(args.arch, device)

    # Prepare the dataset
    root = args.root
    # Normalize the image and noise together
    normalization = preprocess.transforms[-1]
    preprocess_test.transforms.pop(-1)
    preprocess = Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(
                args.image_size, interpolation=InterpolationMode.BICUBIC
            ),
            torchvision.transforms.RandomCrop(args.image_size),
            _convert_image_to_rgb,
            ToTensor(),
        ]
    )
    preprocess_test = Compose(
        [
            torchvision.transforms.Resize(
                args.image_size,
                interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(
                size=(
                    args.image_size,
                    args.image_size)),
            _convert_image_to_rgb,
            ToTensor(),
        ])

    train_set, test_set, text_inputs = load_dataset(args, preprocess, preprocess_test)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )


    # Training setting
    epoch = args.epochs
    lr = args.learning_rate

    # Initialize the prompt
    prompt = Pertubation(args.prompt_size, args.prompt_size, clip_model)
    pad_length = int((224 - args.image_size) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    # Optimizer setting
    for name, p in prompt.named_parameters():
        if "clip" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    param_groups = add_weight_decay(prompt, 0.0, skip_list=("perturbation"))
    optimizer = torch.optim.SGD(param_groups, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch)

    max_acc = 0
    # Begin training
    if not args.evaluate:
        if log_wandb:
            wandb.watch(prompt)

        for e in range(epoch):
            train_loss, train_top1 = train_with_prompt(
                epoch=e,
                train_loader=train_loader,
                prompt=prompt,
                text_inputs=text_inputs,
                pad_dim=pad_dim,
                criterion=criterion,
                optim=optimizer,
                normalization=normalization,
                device=device,
            )

            schedule.step()

            test_acc1, test_acc5 = eval(
                test_loader=test_loader,
                prompt=prompt,
                pad_dim=pad_dim,
                text_inputs=text_inputs,
                normalization=normalization,
                device=device,
            )

            if test_acc1 > max_acc:
                max_acc = test_acc1
                model_state = prompt.state_dict()
                save_dict = {"perturbation": model_state["perturbation"]}
                save_path = args.save_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(save_dict, save_path + "/checkpoint_best.pth")
            print("max acc is {}".format(str(max_acc)))

            if log_wandb:
                log_stauts = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "train_top1": train_top1,
                    "test_acc1": test_acc1,
                    "test_acc5": test_acc5,
                }
                wandb.log(log_stauts, step=e)

    # Begin testing
    else:
        # Load the model
        checkpoint = args.checkpoint
        state_dict = torch.load(checkpoint, map_location="cpu")
        perturbation_state = prompt.state_dict()
        perturbation_state["perturbation"] = state_dict["perturbation"]
        prompt.load_state_dict(perturbation_state)

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_acc1, test_acc5 = eval(
            test_loader=test_loader,
            prompt=prompt,
            pad_dim=pad_dim,
            text_inputs=text_inputs,
            normalization=normalization,
            device=device,
        )

        print("Test acc1 is {}".format(str(test_acc1)))


def train_with_prompt(
    epoch,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    criterion,
    optim,
    normalization,
    device,
):
    start_time = time.time()
    lr = optim.param_groups[0]["lr"]
    all_loss = []
    all_top1 = []
    idx = 0

    for images, labels in tqdm(train_loader):
        # Pad the image
        images = F.pad(images, pad_dim, "constant", value=0)
        images = images.to(device)
        noise = prompt.perturbation.to(device)
        noise = noise.repeat(images.size(0), 1, 1, 1)
        noise.retain_grad()

        images = normalization(images + noise)
        images.require_grad = True

        probs = prompt(images, text_inputs)
        loss = criterion(probs, (labels).to(device))
        loss.backward()

        # update the perturbation
        grad_p_t = noise.grad
        grad_p_t = grad_p_t.mean(0).squeeze(0)
        g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
        scaled_g = grad_p_t / (g_norm + 1e-10)
        scaled_g_pad = scaled_g * prompt.mask.to(device)
        updated_pad = scaled_g_pad * lr
        prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
        prompt.zero_grad()

        all_loss.append(loss.detach().cpu().numpy())
        top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
        all_top1.extend(top1.cpu())
        idx += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(
                np.mean(all_top1)), total_time_str))

    return np.mean(all_loss), np.mean(all_top1)


def eval(test_loader, prompt, pad_dim, text_inputs, normalization, device):
    start_time = time.time()
    all_top1, all_top5 = [], []
    print("starting evaluation")
    for images, labels in tqdm(test_loader):
        with torch.no_grad():
            images = F.pad(images, pad_dim, "constant", value=0)
            images = images.to(device)
            noise = prompt.perturbation.to(device)

            images = normalization(images + noise)
            probs = prompt(images, text_inputs)

            top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
            all_top1.extend(top1.cpu())
            all_top5.extend(top5.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    return np.mean(all_top1), np.mean(all_top5)


class Pertubation(torch.nn.Module):
    def __init__(self, pad_h, pad_w, clip_model):
        super().__init__()
        self.mask = torch.ones((3, 224, 224))
        self.mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0

        delta = torch.zeros((3, 224, 224))
        delta.require_grad = True

        self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)
        self.clip_model = clip_model

    def forward(self, images, text_inputs):
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(text_inputs)
        norm_image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        probs = (
            self.clip_model.logit_scale.exp()
            * norm_image_features
            @ norm_text_features.T
        )

        return probs


if __name__ == "__main__":
    main()
