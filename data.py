import os
import torch
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def mnist():
    path = "data/corruptmnist/"

    train = [load(os.path.join(path, f"train_{x}.npz")) for x in range(0, 5)]
    test = load(os.path.join(path, "test.npz"))

    train_images = np.concatenate(([train[x]["images"] for x in range(len(train))]))
    train_labels = np.concatenate(([train[x]["labels"] for x in range(len(train))]))

    train_images_tensor = torch.Tensor(train_images)
    train_labels_tensor = torch.Tensor(train_labels).type(torch.LongTensor)

    train_data = TensorDataset(train_images_tensor, train_labels_tensor)
    train = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_images_tensor = torch.Tensor(test["images"])
    test_labels_tensor = torch.Tensor(test["labels"]).type(torch.LongTensor)

    test_data = TensorDataset(test_images_tensor, test_labels_tensor)
    test = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train, test
