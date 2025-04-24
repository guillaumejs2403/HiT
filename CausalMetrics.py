import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import os
import tqdm
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from core.model import get_model
from core.datasets import build_dataset
from core.config_base import merge_config

from pytorch_grad_cam import GradCAM


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return ((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)).item()


class CausalMetric():
    def __init__(self, model, steps, device, replace_mode='black'):
        super().__init__()
        self.model = model
        self.steps = steps
        self.device = device
        self.gauss_kernel = gkern(11, 5).to(device)
        self.replace_mode = replace_mode

    def gaussian_filter(self, x):
        return nn.functional.conv2d(x, self.gauss_kernel, padding=11 // 2)

    def replace(self, x1, x2, mask):
        return x1 * mask + x2 * (1 - mask)

    def auc(arr):
        """Returns normalized Area Under Curve of the array."""
        return ((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)).item()

    def produce_mix(self, x, damaged_image, mask):
        NotImplementedError('Function Not Implemented')

    def show_image(self, x):
        x = x.cpu().numpy().transpose((1, 2, 0))
        # Mean and std for ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = std * x + mean
        x = np.clip(x, 0, 1)
        plt.imshow(x)

    def show_map(self, map):
        map = map.cpu().numpy()
        plt.imshow(map, cmap='jet')

    def __call__(self, dataloader, cam_fn, use_probability=True, use_label=False):

        running_scores = torch.zeros(self.steps + 1)
        for x, l in tqdm.tqdm(dataloader, desc='Computing Metric'):
            x = x.to(self.device)
            l = l.to(self.device)
            if use_label:
                class_to_explain = l
            else:
                class_to_explain = self.model(x).argmax(dim=1)
            cam = cam_fn(x, class_to_explain)
            running_scores += self.evaluate_batch(x, cam, class_to_explain, use_probability, verbose=0).sum(dim=0)

        running_scores /= len(dataloader.dataset)
        return auc(running_scores), running_scores

    @torch.no_grad()
    def evaluate_batch(self, x, cam, class_to_explain, use_probability=True, verbose=0):

        r'''
        produces the insertion/deletion curve for each image
        :x: image x \in B,3,H,W
        :cam: \in B,H',W'
        :class_to_explain: \in B,#C
        :verbose: 0=Does not show anything, 1=show last step, 2=show every step
        '''

        # organize
        B, Hp, Wp = cam.shape
        _, C, H, W = x.shape
        _, sort_ids = cam.view(B, -1).sort(dim=1, descending=True)
        sort_ids = torch.arange(0, Hp * Wp, device=self.device, dtype=torch.float).view(1, -1).expand(B, -1).gather(1, sort_ids.argsort(1)).view(B, 1, Hp, Wp)

        ids = F.interpolate(sort_ids, (H, W), mode='nearest').expand(-1, C, -1, -1)
        damaged_image = torch.zeros_like(x) if self.replace_mode == 'black' else self.gaussian_filter(x)
        scores = []

        if verbose in [1, 2]:
            example_scores = []

        for idx, n in enumerate(self.get_sequence(Hp, Wp)):
            pred = self.model(self.produce_mix(x, damaged_image, (ids < n).float()))
            if use_probability:
                pred = torch.softmax(pred, dim=1)
            scores.append(pred[range(B), class_to_explain].view(-1, 1).cpu())

            if verbose in [1, 2]:
                example_scores.append(pred[0, class_to_explain[0]].cpu().numpy())

            if (verbose == 2) or (verbose == 1 and idx == self.steps):
                print('Score {:.2f} at step {}'.format(example_scores[-1], n.item()))
                f = plt.figure()
                plt.subplot(2, 2, 1)
                self.show_image(self.produce_mix(x, damaged_image, (ids < n).float())[0])
                plt.axis('off')
                plt.subplot(2, 2, 2)
                self.show_map(sort_ids[0, 0])
                plt.axis('off')
                plt.title('Ids')
                plt.subplot(2, 2, 3)
                self.show_map(cam[0])
                plt.axis('off')
                plt.title('CAM')
                plt.subplot(2, 2, 4)
                plt.plot(list(range(1, len(example_scores) + 1)), example_scores)
                plt.xlim(0, self.steps + 2)
                plt.ylim(-0.05, 1.05)
                plt.show()
        return torch.cat(scores, dim=1)


class InsertionMetric(CausalMetric):
    def produce_mix(self, x, damaged_image, mask):
        return self.replace(x, damaged_image, mask)

    def get_sequence(self, Hp, Wp):
        return torch.linspace(0, Hp * Wp, self.steps + 1, device=self.device)


class DeletionMetric(CausalMetric):
    def produce_mix(self, x, damaged_image, mask):
        return self.replace(damaged_image, x, mask)

    def get_sequence(self, Hp, Wp):
        return torch.linspace(0, Hp * Wp, self.steps + 1, device=self.device)
        # return torch.linspace(0, Hp * Wp, self.steps + 1, device=self.device).flip(dims=[0,])
    

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
    parser.add_argument('--data-set', default='IMNET', # choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
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
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--compute', default='all', choices=['all', 'ins-b', 'ins-z', 'del-b', 'del-z'])

    return parser.parse_known_args()


def plot_and_save(del_z, del_b, ins_z, ins_b, dataset):
    f = plt.figure()
    plt.plot(del_z, label='Deletion (zeros): {:.2f}'.format(auc(del_z)))
    plt.plot(del_b, label='Deletion (blur): {:.2f}'.format(auc(del_b)))
    plt.plot(ins_z, label='Insertion (zeros): {:.2f}'.format(auc(ins_z)))
    plt.plot(ins_b, label='Insertion (blur): {:.2f}'.format(auc(ins_b)))
    plt.legend()
    plt.savefig(f'ins-del-{dataset}.png')
    plt.close()


def main_worker(rank, world_size, config, distributed):

    # ==========================================
    # CUDA variables
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    torch.set_grad_enabled(False)
    torch.cuda.deterministic = True

    # ==========================================
    # Dataloaders

    testing_set, _ = build_dataset(is_train=False, args=args)
    testing_loader = data.DataLoader(testing_set, sampler=None, shuffle=False,
                                     pin_memory=True, batch_size=args.batch_size)

    # ==========================================
    # Load model

    model, model_kwargs = get_model(config['model']['name'],
                                    config['model']['params'],
                                    '')
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)
    model.eval()
    print('Total parameters: {:.2f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    scores = {
        'del-z': None,
        'del-b': None,
        'ins-z': None,
        'ins-b': None
    }

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
    # Training and testing

    img_idx = 0

    if args.compute == 'all' or args.compute == 'del-z':
        print('Deletion (black)')
        metric = DeletionMetric(
            model=model,
            steps=14 * 14,
            device=device,
            replace_mode='black')
        scores['del-z'], curve_dz = metric(testing_loader, cam_fn, use_probability=True, use_label=False)
    
    if args.compute == 'all' or args.compute == 'del-b':
        print('Deletion (gauss)')
        metric = DeletionMetric(
            model=model,
            steps=14 * 14,
            device=device,
            replace_mode='gauss')
        scores['del-b'], curve_db = metric(testing_loader, cam_fn, use_probability=True, use_label=False)
    
    if args.compute == 'all' or args.compute == 'ins-z':
        print('Insertion (black)')
        metric = InsertionMetric(
            model=model,
            steps=14 * 14,
            device=device,
            replace_mode='black')
        scores['ins-z'], curve_iz = metric(testing_loader, cam_fn, use_probability=True, use_label=False)
    
    if args.compute == 'all' or args.compute == 'ins-b':
        print('Insertion (gauss)')
        metric = InsertionMetric(
            model=model,
            steps=14 * 14,
            device=device,
            replace_mode='gauss')
        scores['ins-b'], curve_ib = metric(testing_loader, cam_fn, use_probability=True, use_label=False)

    if args.compute == 'all':
        print('Deletion AUC (zeros):',scores['del-z'])
        print('Deletion AUC (blur):', scores['del-b'])
        print('Insertion AUC (zeros):', scores['ins-z'])
        print('Insertion AUC (blur):', scores['ins-b'])
        plot_and_save(curve_dz, curve_db, curve_iz, curve_ib, args.data_set)

        pd.DataFrame({'del-z': curve_dz.numpy(),
                      'del-b': curve_db.numpy(),
                      'ins-z': curve_iz.numpy(),
                      'ins-b': curve_ib.numpy()}).to_csv(f'hit-ins-del-{args.data_set}-{args.cam_type}.csv')

    elif args.compute == 'del-z':
        print('Deletion AUC (zeros):',scores['del-z'])
        pd.DataFrame({'del-z': curve_dz.numpy()}).to_csv(f'hit-del-z-{args.data_set}-{args.cam_type}.csv')

    elif args.compute == 'del-b':
        print('Deletion AUC (blur):', scores['del-b'])
        pd.DataFrame({'del-b': curve_db.numpy()}).to_csv(f'hit-del-b-{args.data_set}-{args.cam_type}.csv')

    elif args.compute == 'ins-z':
        print('Insertion AUC (zeros):', scores['ins-z'])
        pd.DataFrame({'ins-z': curve_iz.numpy()}).to_csv(f'hit-ins-z-{args.data_set}-{args.cam_type}.csv')

    elif args.compute == 'ins-b':
        print('Insertion AUC (blur):', scores['ins-b'])
        pd.DataFrame({'ins-b': curve_ib.numpy()}).to_csv(f'hit-ins-b-{args.data_set}-{args.cam_type}.csv')


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
