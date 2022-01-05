import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import MyAwesomeModel
from torch import nn, optim

sns.set()


def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    model = MyAwesomeModel()
    model.train()

    train_set = torch.load("data/processed/corruptmnist/train.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []

    epochs = 2
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch: {e} - Training loss: {running_loss/len(train_set):5f}")
        train_loss.append(running_loss / len(train_set))

        plt.figure()
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.xticks(range(1, epochs + 2))
        plt.ylabel("Loss")
        plt.plot(range(1, args.epochs + 1), train_loss)
        plt.savefig("reports/figures/training.png")
        plt.close()

        torch.save(model.state_dict(), "models/mnist/model.pt")


if __name__ == "__main__":
    train()