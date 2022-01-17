import torch
from torchvision.models import resnet18

model = resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save("deployable_model.pt")
