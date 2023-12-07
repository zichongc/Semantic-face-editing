"""
solving the attribution vector.
"""
import argparse
import os
import torch
import numpy as np
from sklearn import svm

from util import load_dataset


def solve(samples, labels):
    clf = svm.SVC(kernel='linear')
    clf.fit(samples, labels)

    vector = clf.coef_.reshape(1, -1).astype(np.float32)
    return vector / np.linalg.norm(vector), clf


def save(vector, path):
    torch.save(vector, path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default='w')
    parser.add_argument('--attr', type=str, default='Young')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()

    if args.dataset_path is not None:
        dataset_path = args.dataset_path
        xs, ys = load_dataset(path=dataset_path, code=args.code)
    else:
        try:
            dataset_path = f'./datasets/{args.attr}_20000_balanced.pkl'
            xs, ys = load_dataset(path=dataset_path, code=args.code)
        except:
            dataset_path = f'./datasets/{args.attr}_10000_balanced.pkl'
            xs, ys = load_dataset(path=dataset_path, code=args.code)

    print(f'loaded dataset from {dataset_path}.')
    n_samples = xs.shape[0]
    print('solving...')
    direction, classifier = solve(xs[:int(args.ratio*n_samples)], ys[:int(args.ratio*n_samples)])
    direction = torch.tensor(direction)

    output = fr'./checkpoints/attribute_vectors/{args.attr}_{args.code}.pt'
    os.makedirs('./checkpoints/attribute_vectors', exist_ok=True)
    save(direction, path=output)
    print(f'saved solution to {output}.')
