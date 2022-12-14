import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np


def get_index(train_set, model, device):

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4)

    # append every
    num_of_classes = len(train_set.classes)
    d = [[] for x in range(0, num_of_classes)]
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        d[labels].append(images)
    model.eval()

    index = []
    with torch.no_grad():
        for i in range(0, num_of_classes):
            print('now is on the class ', i)
            each_class_images = d[i]
            arg_index = []
            for each_image in each_class_images:
                images = each_image.to(device)
                probs = model(images).cpu()
                arg_index.append(int(torch.argmax(probs, dim=-1)))
            arg_index = np.array(arg_index)
            #arg_index = map(float, arg_index)
            counts = np.bincount(arg_index)

            a, idx1 = torch.sort(torch.tensor(counts), descending=True)

            for j in range(len(idx1)):
                if idx1[j] not in index:
                    index.append(idx1[j])
                    break
                else:
                    continue

    return index
