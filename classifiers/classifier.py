import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms


class Classifier(nn.Module):
    def __init__(self, attr, ckpt_root='./checkpoints/classifiers', device='cuda'):
        super().__init__()
        self.attr = attr
        self.device = device
        self.classifier = self.init_classifier(ckpt_root+f'/{attr}.pth')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_from_sg = transforms.Compose([
            transforms.Normalize(-1, 2),
            self.transform
        ])

    def init_classifier(self, ckpt):
        net = resnet18(pretrained=True)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=2, bias=True)
        torch.nn.init.xavier_uniform_(net.fc.weight)
        # load pretrained model
        net.load_state_dict(torch.load(ckpt), strict=True)
        net = net.to(self.device).eval()
        return net

    def forward(self, img, sg=True):
        img = self.transform_from_sg(img) if sg else self.transform(sg)
        img = img.to(self.device)
        label = self.classifier(img)
        return label
