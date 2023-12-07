import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm

from model import Generator
from e4e_projection import projection as e4e_projection
from face_align import align_face
from manipulation import conditional_manipulate, generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='Beard')
    parser.add_argument('--conditions', type=str, default=None)
    parser.add_argument('--code', type=str, default='w')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nrow', type=int, default=1)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--input', type=str, default='./test_inputs/001.png')
    parser.add_argument('--alpha', type=float, default=3)
    parser.add_argument('--l_alpha', type=float, default=-3)
    parser.add_argument('--r_alpha', type=float, default=3)
    parser.add_argument('--num', type=float, default=10)
    args = parser.parse_args()

    device = args.device
    output = args.output if args.output is not None else './outputs'
    # Load original generator
    generator = Generator(1024, 512, 8, 2).to(device)
    ckpt = torch.load('checkpoints/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=True)
    mean_latent = generator.mean_latent(10000)

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    resize = transforms.Resize((args.size, args.size))
    attr = args.attr

    aligned_face = align_face(args.input, return_tensor=False)
    if os.path.exists('./outputs/inversion_codes/' + os.path.basename(args.input) + '.pt'):
        w = torch.load('./outputs/inversion_codes/' + os.path.basename(args.input) + '.pt')['latent'].unsqueeze(0)
    else:
        w = e4e_projection(aligned_face,
                           './outputs/inversion_codes/' + os.path.basename(args.input) + '.pt', device).unsqueeze(0)

    conditions = args.conditions.split(',') if args.conditions is not None else None
    attr_vectors = {
        a: torch.load(os.path.join('./checkpoints/attribute_vectors', f'{a}_{args.code}.pt')).to(device)
        for a in [args.attr] + ([] if conditions is None else conditions)
    }
    w_edit = conditional_manipulate(w, attr_vectors[args.attr], -args.alpha,
                                    conditions=None if conditions is None else [attr_vectors[i] for i in conditions])

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image = generate(w_edit, generator, input_is_latent=True)
    torchvision.utils.save_image(resize(image),
                                 os.path.join(output, f'{os.path.basename(args.input)}_{attr}_{str(conditions)}.png'),
                                 normalize=True, value_range=(-1, 1), nrow=args.nrow)

    # different degrees
    alphas = np.linspace(args.l_alpha, args.r_alpha, args.num, endpoint=True)
    images = []
    with torch.no_grad():
        for idx, alpha in tqdm.tqdm(enumerate(alphas)):
            w_edit = conditional_manipulate(
                w, attr_vectors[args.attr], -alpha,
                conditions=None if conditions is None else [attr_vectors[i] for i in conditions]
            )
            image = generate(w_edit, generator, input_is_latent=True)
            images.append(image)

    torchvision.utils.save_image(
        resize(torch.cat(images, dim=0)),
        os.path.join(output, f'{os.path.basename(args.input)}_{attr}_{str(conditions)}.png'),
        normalize=True, value_range=(-1, 1), nrow=args.num
    )
