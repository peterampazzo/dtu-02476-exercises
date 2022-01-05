# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import torch
from numpy import load
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn.functional as F


def mnist(input_filepath, output_filepath):
    train = [load(os.path.join(input_filepath, f"train_{x}.npz")) for x in range(0, 5)]
    test = load(os.path.join(input_filepath, "test.npz"))

    train_images = np.concatenate(([train[x]["images"] for x in range(len(train))]))
    train_labels = np.concatenate(([train[x]["labels"] for x in range(len(train))]))

    train_images_tensor = F.normalize(torch.Tensor(train_images))  # normalize
    train_labels_tensor = torch.Tensor(train_labels).type(torch.LongTensor)

    train = TensorDataset(train_images_tensor, train_labels_tensor)

    test_images_tensor = F.normalize(torch.Tensor(test["images"]))
    test_labels_tensor = torch.Tensor(test["labels"]).type(torch.LongTensor)

    test = TensorDataset(test_images_tensor, test_labels_tensor)
    
    torch.save(train, f"{output_filepath}/train.pt")
    torch.save(test, f"{output_filepath}/test.pt")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    mnist(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
