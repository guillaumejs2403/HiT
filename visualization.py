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


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM


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
    parser.add_argument('--cam-type', default='base', choices=['base', 'rollout', 'gradcam'])

    # Misc args
    parser.add_argument('--output-path', default='vis')
    parser.add_argument('--seed', default=None, type=int)

    return parser.parse_known_args()


def save_img(img, cam, pred, label, alpha=0.8, idx=0, output_path=''):
    img = np.array(img)
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(f'label: {label}')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    plt.title(f'pred: {pred}')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, str(idx) + '.png'))
    plt.close()


def main_worker(rank, world_size, config, distributed):

    # ==========================================
    # CUDA variables
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    torch.set_grad_enabled(False)
    torch.cuda.deterministic = True
    os.makedirs(args.output_path, exist_ok=True)

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
    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    print('Total parameters: {:.2f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    # ==========================================
    # Create cam function

    if args.cam_type == 'base':
        def cam_fn(img, class_to_explain):
            _, cam = model.forward_cam(img, interpolation_mode=None)
            return cam[range(cam.shape[0]), class_to_explain]
    elif args.cam_type == 'rollout':
        def cam_fn(img, class_to_explain):
            _, cam = model.forward_rollout(img, interpolation_mode=None)
            return cam
    elif args.cam_type == 'gradcam':

        def reshape_transform(tensor, height=7, width=7):
            if tensor.size(1) == 1:
                return torch.zeros_like(tensor).expand(-1, width * height, -1)
            result = tensor.reshape(tensor.size(0),
                height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        target_layers = [model.encoder.layers[2][-1].ln_1]  # only for hit+pool
        cam_engine = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        
        @torch.enable_grad()
        def cam_fn(img, class_to_explain):
            cam = cam_engine(input_tensor=img)
            cam = torch.tensor(cam, device=img.device)
            return cam

    else:
        raise NotImplementedError(f'CAM type {args.cam_type} not implemented')

    # ==========================================
    # Testing

    mu = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    running_saliency = torch.zeros(args.input_size, args.input_size, device=device)

    img_idx = 0

    print(f'Iteration @ rank {rank}: Validation-{len(testing_loader)}')
    c_batch = 0

    for idx, (img, lab) in enumerate(tqdm.tqdm(testing_loader)):

        img = img.to(device, dtype=torch.float)
        lab = lab.to(device, dtype=torch.long)

        B = img.size(0)

        # with amp.autocast():

        # =====================================
        # computes CAM
        pred = model(img)
        pred = pred.argmax(dim=1)

        cam_pred = cam_fn(img, pred).cpu()
        cam_p = F.relu(cam_pred)
        cam_p = cam_p / cam_p.view(B, -1).max(dim=-1)[0].view(B, 1, 1)  # normalize

        img = torch.clamp(img.cpu() * std + mu, 0, 1)
        img = img.permute((0, 2, 3, 1)).numpy()
        for jdx, (i, c, l, p) in enumerate(zip(img, cam_p.numpy(), lab, pred)):
            save_img(i, c, l.item(), p.item(), idx=c_batch + jdx, output_path=args.output_path)
        c_batch += B

        if c_batch > 100:
            break


if __name__ == '__main__':

    # ==========================================
    # Load arguments and config file

    args, unknowns = arguments()
    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)
    config = merge_config(config, unknowns=unknowns)

    # ==========================================
    # Set up CUDA devices and seed

    if args.gpu_id is not None:
        print(f'USING GPU(S) {args.gpu_id}')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        print('USING ALL GPUS')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    main_worker(0, 1, config, False)
