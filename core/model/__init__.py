import torch
import timm

from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer

from core.model.utils import interpolate_embeddings
from core.model.IT import (
    InterpretableVisualTransformer,
    SpatialInterpretableTransformer
)


def get_model(model_name, model_args, weights=''):
    args_for_training = {}
    if model_name == 'vit':
        print('Loading ViT')
        model = VisionTransformer(**model_args)
        if weights:
            print('Loading weights from', weights)
            arch_name, version = weights.split('.')
            weights = getattr(models, arch_name)
            weights = getattr(weights, version)
            # we expect a ViT model, so we remove the interpolate positional
            # embeddings and linear head
            sd = interpolate_embeddings(image_size=model_args['image_size'],
                                        patch_size=model_args['patch_size'],
                                        model_state=weights.get_state_dict(),
                                        reset_heads=True,
                                        reset_conv_proj=True)
            model.load_state_dict(sd, strict=False)

    elif model_name == 'hit-no-pool':
        print('Loading HiT')
        model = InterpretableVisualTransformer(**model_args)
        if weights:
            print('Loading weights from', weights)
            try:
                print('Trying to load from torchvision...')
                # if models has the weights, then we can load them
                arch_name, version = weights.split('.')
                weights = getattr(models, arch_name)
                weights = getattr(weights, version)
                # we expect a ViT model, so we remove the interpolate positional
                # embeddings and linear head
                sd = interpolate_embeddings(image_size=model_args['image_size'],
                                            patch_size=model_args['patch_size'],
                                            model_state=weights.get_state_dict(),
                                            reset_heads=True,
                                            reset_conv_proj=False)
                model.load_state_dict(sd, strict=False)
            except:
                print('Did not found a torchvision weights, trying loading it as a ".pth" object')
                # else, it should be a torch dict
                sd = interpolate_embeddings(image_size=model_args['image_size'],
                                            patch_size=model_args['patch_size'],
                                            model_state=torch.load(weights),
                                            reset_heads=True,
                                            reset_conv_proj=False)
                model.load_state_dict(sd, strict=False)

    elif model_name == 'hit':
        print('Loading Spatial HiT')
        model = SpatialInterpretableTransformer(**model_args)
        if weights:
            print('Loading weights from', weights)
            try:
                print('Trying to load from torchvision...')
                # if models has the weights, then we can load them
                arch_name, version = weights.split('.')
                weights = getattr(models, arch_name)
                weights = getattr(weights, version)
                # we expect a ViT model, so we remove the interpolate positional
                # embeddings and linear head
                sd = interpolate_embeddings(image_size=model_args['image_size'],
                                            patch_size=model_args['patch_size'],
                                            model_state=weights.get_state_dict(),
                                            reset_heads=True,
                                            reset_conv_proj=False)
                model.load_state_dict(sd, strict=False)
            except:
                print('Did not found a torchvision weights, trying loading it as a ".pth" object')
                # else, it should be a torch dict
                sd = interpolate_embeddings(image_size=model_args['image_size'],
                                            patch_size=model_args['patch_size'],
                                            model_state=torch.load(weights),
                                            reset_heads=True,
                                            reset_conv_proj=False)
                model.load_state_dict(sd, strict=False)

    elif model_name == 'timm':

        # for timm models
        # model_args['pretrained']: bool
        # model_args['model_name']: str
        # model_args['num_classes']: int to change classes, should work fine
        print('Loading model', model_args['model_name'], 'from TIMM')
        model = timm.create_model(**model_args)

    else:
        print('Getting the model from torchvision...')
        model = getattr(models, model_name)(**model_args)
        if weights:
            weights = getattr(models, arch_name)
            weights = getattr(weights, version)
            model.load_state_dict(weights.get_state_dict())

    print('Load done')
    return model, args_for_training
