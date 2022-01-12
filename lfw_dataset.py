"""
LFW dataloading
"""
import argparse
import time

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.transform = transform
        self.data = []

        for subdir, dirs, files in os.walk(path_to_folder):
            for file in files:
                self.data.append(os.path.join(subdir,file))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.data[index]) as img:
            img.show()
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        pass
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')