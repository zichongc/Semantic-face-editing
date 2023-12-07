import argparse
import torch
import pickle
import numpy as np


@torch.no_grad()
def load_model(generator, model_file_path):
    ckpt = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    return generator.mean_latent(50000)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_dataset(path, code):
    with open(path, 'rb') as file:
        dataset = pickle.load(file)

    xs, ys = [], []
    for i in range(len(dataset.keys())):
        xs.append(dataset[i][code])
        ys.append(dataset[i]['label'])

    xs = np.concatenate(xs, axis=0)
    ys = np.array(ys)
    return xs, ys
