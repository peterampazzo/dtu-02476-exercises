import os
from dotenv import load_dotenv
import torch
from model import MyAwesomeModel
from torch import nn, optim
import hydra
from omegaconf import OmegaConf

load_dotenv()

@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    DIR = os.getenv('DIRECTORY')
    print("Training day and night")
   
    model = MyAwesomeModel()

    train = torch.load(f"{DIR}data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.params.lr)

    train_loss = []

    for e in range(config.params.epochs):
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