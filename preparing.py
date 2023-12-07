"""
Creating paired data (latent_code, attr_class) for solving attribution vector.
"""
import os
import pickle
import argparse
import warnings
import torch
import torch.nn as nn

from model import Generator
from classifiers.classifier import Classifier
warnings.filterwarnings('ignore')


class Labeler(nn.Module):
    def __init__(self, classifier, device='cuda'):
        super().__init__()
        self.device = device
        self.classifier = classifier.to(device)

    def forward(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            label = self.classifier(img)
        return label


class Decoder(nn.Module):
    def __init__(self, ckpt_path=r'./checkpoints/stylegan2-ffhq-config-f.pt', device='cuda'):
        super().__init__()
        self.device = device
        self.G = Generator(1024, 512, 8, 2).to(device)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(ckpt["g_ema"], strict=True)
        self.mean_latent = self.G.mean_latent(10000)

    def get_latent(self, noise):
        noise = noise.to(self.device)
        with torch.no_grad():
            w_code = self.G.get_latent(noise, truncation_latent=self.mean_latent, truncation=0.7)
        return w_code

    def forward(self, code, input_is_w=True):
        with torch.no_grad():
            if not input_is_w:
                code = code.to(self.device)
                code = self.G.get_latent(code, truncation_latent=self.mean_latent, truncation=0.7)
            code = code.unsqueeze(1).repeat(1, self.G.n_latent, 1)
            img = self.G(code, input_is_latent=True)
        return img


def create_dataset(n_samples, dataset_path, device='cuda'):
    classifier = Classifier(attr=args.attr)
    labeler = Labeler(classifier)
    decoder = Decoder(device=device)
    dataset = {}
    pos, neg, idx = 0, 0, 0

    while pos < n_samples // 2 or neg < n_samples // 2:
        z_code = torch.randn([1, 512], device=device)
        w_code = decoder.get_latent(z_code)
        image = decoder(w_code)
        label = labeler(image)
        predict = torch.softmax(label, dim=1)[0]
        if pos < n_samples // 2 and predict[0] > predict[1] and predict[0] >= 0.75:
            dataset[idx] = {'z': z_code.cpu().numpy(), 'w': w_code.cpu().numpy(), 'label': 1}
            pos += 1
            idx += 1
        elif neg < n_samples // 2 and predict[1] > predict[0] and predict[1] >= 0.75:
            dataset[idx] = {'z': z_code.cpu().numpy(), 'w': w_code.cpu().numpy(), 'label': 0}
            neg += 1
            idx += 1

        if min(pos, neg) % 10 == 0 and min(pos, neg) != 0:
            print(f'collected {pos+neg} samples, in which there are {pos} positive samples and {neg} negative samples.')

    with open(dataset_path, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(path):
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create dataset.")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--attr', type=str, default='Chubby',
                        help='if None, create a new dataset with (z, w) codes containing n_samples.')
    parser.add_argument('--n_samples', type=int, default=20000)
    parser.add_argument('--output', type=str, default=r'./datasets')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output = os.path.join(args.output, f'{args.attr}_{args.n_samples}_balanced.pkl')

    create_dataset(args.n_samples, output)
