import argparse
import sys
import os
from dotenv import load_dotenv
import torch
from model import MyAwesomeModel
from torch import nn, optim

load_dotenv()


def train():
    DIR = os.getenv('DIRECTORY')
    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    model = MyAwesomeModel()

    train = torch.load(f"{DIR}data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []

    epochs = 2
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch: {e} - Training loss: {running_loss/len(train_set):5f}")
        train_loss.append(running_loss / len(train_set))


if __name__ == "__main__":
    train()