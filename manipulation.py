import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from model import Generator


def manipulate(latent, attr_vec, alpha):
    latent = latent + alpha * attr_vec
    return latent


def conditional_manipulate(latent, attr_vec, alpha, conditions=None):
    assert len(attr_vec.shape) == 2 and attr_vec.shape[0] == 1

    if conditions is None:
        return latent + alpha * attr_vec

    if len(conditions) == 1:
        cond = conditions[0]
        assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
                cond.shape[1] == attr_vec.shape[1])
        new = attr_vec - (attr_vec[0].dot(cond[0])).unsqueeze(0) * cond
        conditioned_attr_vec = new / torch.norm(new)

    elif len(conditions) == 2:
        cond_1 = conditions[0]
        cond_2 = conditions[1]
        assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and cond_1.shape[1] == attr_vec.shape[1])
        assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and cond_2.shape[1] == attr_vec.shape[1])
        primal_cond_1 = (attr_vec[0].dot(cond_1[0])).unsqueeze(0)
        primal_cond_2 = (attr_vec[0].dot(cond_2[0])).unsqueeze(0)
        cond_1_cond_2 = (cond_1[0].dot(cond_2[0])).unsqueeze(0)
        alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
                1 - cond_1_cond_2 ** 2 + 1e-8)
        beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
                1 - cond_1_cond_2 ** 2 + 1e-8)
        new = attr_vec - alpha * cond_1 - beta * cond_2
        conditioned_attr_vec = new / torch.norm(new)

    else:
        for cond_boundary in conditions:
            assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
                    cond_boundary.shape[1] == attr_vec.shape[1])
        cond_boundaries = torch.cat(conditions, dim=0)
        a = torch.matmul(cond_boundaries, cond_boundaries.T)
        b = torch.matmul(cond_boundaries, attr_vec.T)
        x = np.linalg.solve(a.cpu().numpy(), b.cpu().numpy())
        new = attr_vec.cpu().numpy() - (np.matmul(x.T, cond_boundaries.cpu().numpy()))
        conditioned_attr_vec = torch.tensor(new / np.linalg.norm(new), device=attr_vec.device)

    return latent + alpha * conditioned_attr_vec


def generate(code, g, input_is_w=False, input_is_latent=False):
    assert (input_is_latent and input_is_w) is False
    with torch.no_grad():
        if input_is_latent:
            assert len(code.shape) == 3
            return g(code, input_is_latent=True)
        if not input_is_w and not input_is_latent:
            code = g.get_latent(code, truncation_latent=mean_latent, truncation=0.7)
        code = code.unsqueeze(1).repeat(1, g.n_latent, 1)
        img = g(code, input_is_latent=True)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='Beard')
    parser.add_argument('--conditions', type=str, default=None)
    parser.add_argument('--code', type=str, default='w')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nrow', type=int, default=3)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--n_each', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=3.)
    args = parser.parse_args()

    device = args.device
    # Load original generator
    generator = Generator(1024, 512, 8, 2).to(device)
    ckpt = torch.load('checkpoints/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=True)
    mean_latent = generator.mean_latent(10000)

    resize = transforms.Resize((args.size, args.size))

    seed = args.seed
    torch.manual_seed(seed)

    attr = args.attr
    attr_vector = torch.load(os.path.join('./checkpoints/attribute_vectors', f'{attr}_{args.code}.pt')).to(device)
    output = f'./outputs' if args.output is None else args.output
    os.makedirs(output, exist_ok=True)

    with torch.no_grad():
        generator.eval()

        z = torch.randn(1, 512, device=device)
        w = generator.get_latent(z, truncation_latent=mean_latent, truncation=0.7)
        origin = generate(w, generator, input_is_w=True)

        # single attribute editing (tes
        w_edit = manipulate(w, attr_vec=attr_vector, alpha=args.alpha)
        edited1 = generate(w_edit, generator, input_is_w=True)
        w_edit = manipulate(w, attr_vec=attr_vector, alpha=-args.alpha)
        edited2 = generate(w_edit, generator, input_is_w=True)

        torchvision.utils.save_image(resize(torch.cat([edited1, origin, edited2], dim=0)),
                                     os.path.join(output, f'{attr}_{seed}.png'),
                                     normalize=True, value_range=(-1, 1), nrow=args.nrow)

        # # conditional editing (test)
        # attr_vectors = {
        #     a: torch.load(os.path.join('./checkpoints/attribute_vectors', f'{a}_{args.code}.pt')).to(device)
        #     for a in ['Hat', 'Smiling', 'Male', 'Young']
        # }
        # w_smile = manipulate(w, attr_vectors['Smiling'], -2)
        # img_smile = generate(w_smile, generator, input_is_w=True)
        # w_smile_young = manipulate(w_smile, attr_vectors['Young'], 2)
        # img_smile_young = generate(w_smile_young, generator, input_is_w=True)
        # w_smile_young_cond = conditional_manipulate(w_smile, attr_vectors['Young'], 2,
        #                                             conditions=[attr_vectors['Smiling'],
        #                                                         attr_vectors['Male'],
        #                                                         attr_vectors['Hat']])
        # img_smile_young_cond = generate(w_smile_young_cond, generator, input_is_w=True)
        #
        # torchvision.utils.save_image(resize(torch.cat([img_smile, img_smile_young, img_smile_young_cond], dim=0)),
        #                              os.path.join(output, f'sy_{seed}.png'),
        #                              normalize=True, value_range=(-1, 1), nrow=args.nrow)
