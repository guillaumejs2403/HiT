import os
import cv2
import tqdm
import yaml
import random
import argparse
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp

from core.model import get_model
from core.datasets import build_dataset
from core.config_base import merge_config


def arguments():
    parser = argparse.ArgumentParser(description='Training routine.')

    parser.add_argument('--gpu-id', '-g', default=None, type=str,
                        help='GPU id''s')
    parser.add_argument('--config-file', default='cifar.yaml', type=str,
                        help='Path to configuration file')
    parser.add_argument('--weights', default='', type=str,
                        help='Model weights')

    # Dataset args
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', #choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

    # Model args
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    return parser.parse_known_args()


def main_worker(rank, world_size, config, distributed):

    # ==========================================
    # CUDA variables
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    torch.set_grad_enabled(False)

    # ==========================================
    # Dataloaders

    testing_set, _ = build_dataset(is_train=False, args=args)
    testing_loader = data.DataLoader(testing_set, sampler=None, shuffle=True,
                                     pin_memory=True, batch_size=args.batch_size)

    # ==========================================
    # Load model and loss

    model, model_kwargs = get_model(config['model']['name'],
                                    config['model']['params'],
                                    '')
    state_dict = torch.load(args.weights, map_location='cpu')['model']
    # state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict().keys()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    # ==========================================
    # Training and testing

    mu = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    img_idx = 0
    acc = np.zeros(11 + 11 + 1)
    acc_removed = np.zeros(12)
    acc_added = np.zeros(12)
    layers_to_remove = list(range(12))

    print(f'Iteration @ rank {rank}: Validation-{len(testing_loader)}')
    cam_layer = torch.zeros(12)
    for idx, (img, lab) in enumerate(tqdm.tqdm(testing_loader)):

        img = img.to(device, dtype=torch.float)
        lab = lab.to(device, dtype=torch.long)

        B = img.size(0)

        with amp.autocast():

            # =====================================
            # computes CAM
            pred, cam = model.forward_cam_layer(img.detach())
            pred = pred.argmax(dim=1)

            acc[0] += (pred == lab).sum().item()

            # adds from the first layer the the last one
            for idx in range(1, 12):
                pred = model.forward_without_layers(img, layers_to_remove=layers_to_remove[idx:]).argmax(dim=1)
                acc[idx] += (pred == lab).sum().item()

            # remove from layer 0 to layer 11
            for idx in range(1, 12):
                pred = model.forward_without_layers(img, layers_to_remove=layers_to_remove[:idx]).argmax(dim=1)
                acc[11 + idx] += (pred == lab).sum().item()

            # with a single layer
            for idx in range(0, 12):
                pred = model.forward_without_layers(img, layers_to_remove=list(range(0, idx)) + list(range(idx + 1, 12))).argmax(dim=1)
                acc_added[idx] += (pred == lab).sum().item()

            # all except one layer
            for idx in range(0, 12):
                pred = model.forward_without_layers(img, layers_to_remove=[idx]).argmax(dim=1)
                acc_removed[idx] += (pred == lab).sum().item()

            cam_pred = cam[range(B), pred].cpu()
            cam_layer += cam_pred.sum(dim=0)

    print('Layer saliency', cam_layer / len(testing_set))
    print('Acc:', acc / len(testing_set))
    print('Without Layer:', acc_removed / len(testing_set))
    print('Only with Layer:', acc_added / len(testing_set))


if __name__ == '__main__':

    # ==========================================
    # Load arguments and config file

    args, unknowns = arguments()
    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)
    config = merge_config(config, unknowns=unknowns)

    # os.makedirs(os.path.join(config['output_dir'], 'vis'), exist_ok=True)

    # ==========================================
    # Set up CUDA devices and seed

    if args.gpu_id is not None:
        print(f'USING GPU(S) {args.gpu_id}')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        print('USING ALL GPUS')

    main_worker(0, 1, config, False)
