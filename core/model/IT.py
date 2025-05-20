import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import MLP
from torchvision.models.vision_transformer import MLPBlock

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from core.model.modules.attention import MultiheadAttention

from timm.models.layers import DropPath


# ========================================================
# Transformer Main Opt
# ========================================================


class EncoderBlockBase(nn.Module):
    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_ratio: int,
                 dropout: float,
                 attention_dropout: float,
                 mlp_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 drop_path: float = 0.0,
                 remove_mlp: bool = False):
        super().__init__()
        self.num_heads = num_heads
        mlp_dim = int(mlp_ratio * hidden_dim)

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) if drop_path > 0. else nn.Identity()

        # MLP block. This is useful for distributed dataparallel Only used in HiT architecture
        if not remove_mlp:
            self.ln_2 = norm_layer(hidden_dim)
            self.mlp = MLPBlock(hidden_dim, mlp_dim, mlp_dropout)
        else:
            print('Removing last MLP block (and layer drop is applicable) since it will not be used')
            self.mlp = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


class EncoderBlockViT(EncoderBlockBase):
    def forward(self,
                input: torch.Tensor,
                class_token: torch.Tensor,
                get_additional_outputs: bool = False):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = torch.cat((class_token, input), dim=1)
        y = self.ln_1(x)

        y, attn_map, vectors = self.self_attention(y, y, y, need_weights=get_additional_outputs)
        x = x + self.drop_path(self.dropout(y))
        
        if self.mlp is not None:
            x = x + self.drop_path(self.dropout(self.mlp(self.ln_2(x))))

        other_outputs = {'attn_map': attn_map,
                         'vectors': vectors}

        return x[:, 1:], x[:, :1], other_outputs


class EncoderBlockHiT(EncoderBlockBase):
    def forward(self,
                input: torch.Tensor,
                class_token: torch.Tensor,
                get_additional_outputs: bool = False):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        kv = self.ln_1(input)
        q = self.ln_1(class_token)

        q, attn_map, vectors = self.self_attention(q, kv, kv, need_weights=get_additional_outputs)
        class_token = class_token + self.drop_path(self.dropout(q))

        if self.mlp is not None:
            input = self.drop_path(self.dropout(self.mlp(self.ln_2(input)))) \
                    + input

        other_outputs = {'attn_map': attn_map,
                         'vectors': vectors}

        return input, class_token, other_outputs


def get_block_version(version: str = 'hit'):
    print('Block version:', version)
    if version == 'hit':
        return EncoderBlockHiT
    elif version == 'vit':
        return EncoderBlockViT


# ========================================================
# Encoder and Backbone
# ========================================================


class Encoder(nn.Module):
    def __init__(self,
                 seq_length,
                 num_layers,
                 num_heads,
                 hidden_dim,
                 mlp_ratio,
                 dropout,
                 attention_dropout,
                 mlp_dropout,
                 drop_path,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 block_version: str = '2',
                 ln_at_the_end: bool = True):
        super().__init__()
        self.pos_embedding = None
        if seq_length != -1:
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.dropout = nn.Dropout(dropout)
        EncoderBlock = get_block_version(block_version)
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
                norm_layer=norm_layer,
                drop_path=drop_path,
                remove_mlp=block_version == '2.2' and (i == (num_layers - 1))  # last layer is not used in hit
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim) if ln_at_the_end else None

    def forward(self, x, class_token, get_additional_outputs=False,
                layer_outputs=[]):
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, 1:]
            class_token = class_token + self.pos_embedding[:, :1]

        tokens = [class_token]
        vectors = []
        attn_maps = []

        for idx, module in enumerate(self.layers):
            x, class_token, other_outputs = module(
                input=x,
                class_token=class_token,
                get_additional_outputs=get_additional_outputs or idx in layer_outputs
            )

            if get_additional_outputs or idx in layer_outputs:
                vectors.append(other_outputs['vectors'])
                attn_maps.append(other_outputs['attn_map'])
                tokens.append(class_token)

        # store the attention for future use
        if get_additional_outputs or len(layer_outputs) != 0:
            self.class_tokens = torch.cat(tokens, dim=1)
            self.vectors = torch.cat(vectors, dim=1)
            self.attn_maps = torch.cat(attn_maps, dim=1)
    
        if self.ln is not None:
            class_token = self.ln(class_token)
        return x, class_token


class InterpretableVisualTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_ratio: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 mlp_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 num_classes: int = 1000,
                 use_pos_tokens: bool = True,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 block_version: str = '2.3',
                 ln_at_the_end: bool = True):
        super().__init__()

        seq_length = (image_size // patch_size) ** 2 + 1

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = Encoder(seq_length=seq_length if use_pos_tokens else -1,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_dim=hidden_dim,
                               mlp_ratio=mlp_ratio,
                               dropout=dropout,
                               attention_dropout=attention_dropout,
                               mlp_dropout=mlp_dropout,
                               drop_path=drop_path,
                               norm_layer=norm_layer,
                               block_version=block_version,
                               ln_at_the_end=ln_at_the_end)

        # we create the blocks with the same names than the vits to transfer the weights
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers['head'] = nn.Linear(hidden_dim, num_classes, bias=True)
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.heads = nn.Sequential(heads_layers)

        # initialization like in ViT
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        '''
        process the input image into the patches

        Extracted from
        https://github.com/pytorch/vision/blob/15c166ac127db5c8d1541b3485ef5730d34bb68a/torchvision/models/vision_transformer.py#L268C5-L287C17
        '''
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, return_features=False, get_additional_outputs=False):
        # patchifying the image. B,3,H,W -> B,(n_w * n_y),C
        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)  # 1,1,C -> B,1,C
        x, class_token = self.encoder(x=x,
                                      class_token=class_token,
                                      get_additional_outputs=get_additional_outputs,
                                      layer_outputs=[])
        class_token = class_token.squeeze(dim=1)  # should be B,1,C -> B,C
        if return_features:
            return self.heads(class_token), x 
        return self.heads(class_token)

    def forward_cam(self, x, interpolation_mode='nearest', after_softmax=False):

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = image_size // patch_size

        # compute prediction
        logits = self(x, get_additional_outputs=True)

        # =====================================
        # extract patches before the linear layer, this is exactly as in
        # PrototypeAugmentedInterpretableVisualTransformer.extract_sa_features_v2
        all_biases = []
        for module in self.encoder.layers:
            all_biases.append(module.self_attention.out_proj.bias.view(1, 1, 1, -1))
        # shape [1, Layer, 1, feat_dim]
        all_biases = torch.cat(all_biases, dim=1)
        all_biases = all_biases.expand(self.encoder.vectors.size(0), -1, -1, -1)

        # shape [B, Layer, Seq Lenght, feat_dim], including the cls if applicable
        # to extract the cls when used, it should be tokens[:, :, 0, :]
        tokens = self.encoder.vectors
        tokens = tokens + all_biases / tokens.size(2)  # broadcast on dimension 0 and 2
        seq_length = tokens.size(2)
        
        class_token = self.class_token.expand(self.encoder.vectors.size(0), seq_length, -1)  # [1,1,hidden_dim] > [B,seq_length,hidden_dim]

        # sum over layer dimension to get the final sequence
        # the cls and spatial tokens will be broken in the
        # classification head. Divide by seq_length to evenly distribute the CLS token
        # [B, Seq Lenght, feat_dim]
        tokens = tokens.sum(dim=1) + class_token / seq_length
        if self.encoder.ln is not None:
            # manual layernorm
            weight = self.encoder.ln.weight.view(1, 1, tokens.size(2))  # [768] -> [1,1,768]
            bias = self.encoder.ln.bias.view(1, 1, tokens.size(2)) / tokens.size(1)
            cls_tokens = tokens.sum(dim=1, keepdim=True)  # [B,Seq Lenght,feat_dim] -> [B,1,768]

            # compute stats
            mean = cls_tokens.mean(dim=2, keepdim=True) / tokens.size(1)  # [B,1,768] -> [B,1,1]
            var = cls_tokens.var(dim=2, correction=0, keepdim=True)  # [B,1,768] -> [B,1,1]
            tokens = (tokens - mean) / (var + self.encoder.ln.eps).sqrt()  # [B,Seq Lenght,feat_dim] -> [B,Seq Lenght,feat_dim]
            tokens = tokens * weight + bias

        # cam = F.linear(tokens, self.heads[0].weight, bias=self.heads[0].bias)
        cam = F.linear(tokens, self.heads[0].weight, bias=None)
        # cam = self.heads(tokens)
        cam = cam.permute((0, 2, 1))
        cam = cam.view(B, num_classes, num_patches, num_patches)
        if after_softmax:
            cam = F.softmax(cam, dim=1)
        if interpolation_mode is None:
            return logits, cam
        return logits, F.interpolate(cam, size=(image_size, image_size), mode=interpolation_mode)

    def forward_rollout(self, x, interpolation_mode=None, after_softmax=False):

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = image_size // patch_size

        # compute prediction
        logits = self(x, get_additional_outputs=True)

        # self.encoder.attn_maps -> B,#heads,C
        cam = self.encoder.attn_maps.mean(dim=1)
        cam = cam.reshape(B, num_patches, num_patches)

        return logits, cam

    def forward_cam_layer(self, x):

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = image_size // patch_size

        # compute prediction
        logits = self(x, get_additional_outputs=True)

        # =====================================
        # extract patches before the linear layer, this is exactly as in
        # PrototypeAugmentedInterpretableVisualTransformer.extract_sa_features_v2
        all_biases = []
        for module in self.encoder.layers:
            all_biases.append(module.self_attention.out_proj.bias.view(1, 1, 1, -1))
        # shape [1, Layer, 1, feat_dim]
        all_biases = torch.cat(all_biases, dim=1)
        all_biases = all_biases.expand(self.encoder.vectors.size(0), -1, -1, -1)

        # shape [B, Layer, Seq Lenght, feat_dim], including the cls if applicable
        # to extract the cls when used, it should be tokens[:, :, 0, :]
        tokens = self.encoder.vectors
        tokens = tokens + all_biases / tokens.size(2)  # broadcast on dimension 0 and 2
        seq_length = tokens.size(2)
        num_layers = tokens.size(1)
        
        class_token = self.class_token.expand(self.encoder.vectors.size(0), num_layers, -1)  # [1,1,hidden_dim] > [B,seq_length,hidden_dim]

        # sum over layer dimension to get the final sequence
        # the cls and spatial tokens will be broken in the
        # classification head. Divide by seq_length to evenly distribute the CLS token
        # [B, Layer, feat_dim]
        tokens = tokens.sum(dim=2) + class_token / num_layers

        if self.encoder.ln is not None:
            # manual layernorm
            weight = self.encoder.ln.weight.view(1, 1, tokens.size(2))  # [768] -> [1,1,768]
            bias = self.encoder.ln.bias.view(1, 1, tokens.size(2)) / tokens.size(1)
            cls_tokens = tokens.sum(dim=1, keepdim=True)  # [B,Seq Lenght,feat_dim] -> [B,1,768]

            # compute stats
            mean = cls_tokens.mean(dim=2, keepdim=True) / tokens.size(1)  # [B,1,768] -> [B,1,1]
            var = cls_tokens.var(dim=2, correction=0, keepdim=True)  # [B,1,768] -> [B,1,1]
            tokens = (tokens - mean) / (var + self.encoder.ln.eps).sqrt()  # [B,Seq Lenght,feat_dim] -> [B,Seq Lenght,feat_dim]
            tokens = tokens * weight + bias

        cam = F.linear(tokens, self.heads[0].weight, bias=None)
        # cam = self.heads(tokens)
        return logits, cam

    def forward_without_layers(self, x, layers_to_remove=[]):

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = image_size // patch_size

        # compute prediction
        _ = self(x, get_additional_outputs=True)

        # =====================================
        # extract patches before the linear layer, this is exactly as in
        # PrototypeAugmentedInterpretableVisualTransformer.extract_sa_features_v2
        all_biases = []
        for module in self.encoder.layers:
            all_biases.append(module.self_attention.out_proj.bias.view(1, 1, 1, -1))
        # shape [1, Layer, 1, feat_dim]
        all_biases = torch.cat(all_biases, dim=1)
        all_biases = all_biases.expand(self.encoder.vectors.size(0), -1, -1, -1)

        # shape [B, Layer, Seq Lenght, feat_dim], including the cls if applicable
        # to extract the cls when used, it should be tokens[:, :, 0, :]
        tokens = self.encoder.vectors
        tokens = tokens + all_biases / tokens.size(2)  # broadcast on dimension 0 and 2
        seq_length = tokens.size(2)
        num_layers = tokens.size(1)

        layers_to_keep = [a for a in range(num_layers) if a not in layers_to_remove]
        
        class_token = self.class_token.expand(self.encoder.vectors.size(0), len(layers_to_keep), -1)  # [1,1,hidden_dim] > [B,seq_length,hidden_dim]

        # sum over layer dimension to get the final sequence
        # the cls and spatial tokens will be broken in the
        # classification head. Divide by seq_length to evenly distribute the CLS token
        # [B, Layer, feat_dim]
        tokens = tokens.sum(dim=2) 
        tokens = tokens[:, layers_to_keep]
        tokens = tokens + class_token / tokens.size(1)
        tokens = tokens.sum(dim=1)
        return self.heads(self.encoder.ln(tokens))


# =========================================================================
# =========================================================================
#
# Spatial Pat/HiT
#
# =========================================================================
# =========================================================================


class PoolingTokens(nn.Module):
    def __init__(self, factor, in_dim, out_dim):
        super().__init__()
        self.factor = factor
        self.avg_pool = nn.AvgPool2d(kernel_size=factor)
        self.linear = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self,
                input,
                class_token,
                get_additional_outputs=False):
        '''
        :input: token tensor
        :class_token: cls
        :get_additional_outputs: Does nothing
        '''

        if self.factor == 1:
            if isinstance(self.linear, nn.Identity):
                return input, class_token, None
            else:
                return self.linear(input), self.linear(class_token), None

        B, S, C = input.shape
        sS = int(math.sqrt(S))
        input = input.permute((0, 2, 1)).view(B, C, sS, sS)
        input = self.avg_pool(input)
        input = input.view(B, C, -1).permute(0, 2, 1)

        return self.linear(input), self.linear(class_token), None


class SpatialEncoder(nn.Module):
    def __init__(self,
                 seq_length,
                 num_heads,
                 mlp_ratio,
                 dropout,
                 attention_dropout,
                 mlp_dropout,
                 drop_path,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 block_version: str = '2',
                 ln_at_the_end: bool = True,
                 blocks: list = [4, 4, 12, 4],
                 block_dims: list = [64, 128, 320, 512]):
        super().__init__()
        self.pos_embedding = None
        if seq_length != -1:
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, block_dims[0]).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        EncoderBlock = get_block_version(block_version)

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        last_block_hidden_dim = block_dims[0]
        for idx, (num_layers, hidden_dim) in enumerate(zip(blocks, block_dims)):
            layer = OrderedDict()
            layer['pool'] = PoolingTokens(2 if idx != 0 else 1, last_block_hidden_dim, hidden_dim)
            last_block_hidden_dim = hidden_dim
            for jdx in range(num_layers):
                layer[f'encoder_layer_{jdx}'] = EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    remove_mlp=block_version == '2.2' and (jdx == (num_layers - 1)) and (idx == (len(blocks) - 1)),
                )
            layers[f"block_{idx}"] = nn.Sequential(layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim) if ln_at_the_end else None

    def forward(self, x, class_token, get_additional_outputs=False):

        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, 1:]
            class_token = class_token + self.pos_embedding[:, :1]

        for idx, module in enumerate(self.layers):
            for jdx, m in enumerate(module):
                x, class_token, _ = m(
                    input=x,
                    class_token=class_token,
                    get_additional_outputs=False
                )

        if self.ln is not None:
            class_token = self.ln(class_token)
        return x, class_token


class SpatialInterpretableTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 num_heads: int,
                 mlp_ratio: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 mlp_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 num_classes: int = 1000,
                 use_pos_tokens: bool = True,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 block_version: str = '2.3',
                 ln_at_the_end: bool = True,
                 blocks=[4, 4, 12, 4],
                 block_dims=[64, 128, 320, 512]):
        super().__init__()

        seq_length = (image_size // patch_size) ** 2 + 1

        self.class_token = nn.Parameter(torch.zeros(1, 1, block_dims[0]))

        self.encoder = SpatialEncoder(
            seq_length=seq_length if use_pos_tokens else -1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            drop_path=drop_path,
            norm_layer=norm_layer,
            block_version=block_version,
            ln_at_the_end=ln_at_the_end,
            blocks=blocks,
            block_dims=block_dims,
        )

        # we create the blocks with the same names than the vits to transfer the weights
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers['head'] = nn.Linear(block_dims[-1], num_classes, bias=True)
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=block_dims[0],
                                   kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = block_dims[0]
        self.num_classes = num_classes

        self.heads = nn.Sequential(heads_layers)

        # initialization like in ViT
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        '''
        process the input image into the patches

        Extracted from
        https://github.com/pytorch/vision/blob/15c166ac127db5c8d1541b3485ef5730d34bb68a/torchvision/models/vision_transformer.py#L268C5-L287C17
        '''
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, return_features=False, get_additional_outputs=False):
        # patchifying the image. B,3,H,W -> B,(n_w * n_y),C
        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)  # 1,1,C -> B,1,C
        x, class_token = self.encoder(x=x,
                                      class_token=class_token,
                                      get_additional_outputs=get_additional_outputs)
        class_token = class_token.squeeze(dim=1)  # should be B,1,C -> B,C
        if return_features:
            return self.heads(class_token), x
        return self.heads(class_token)

    def forward_cam(self, x, interpolation_mode=None, after_softmax=False):

        # include here layer dimension increase

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = (image_size // patch_size) ** 2
        # =====================================
        # Virtually, manual forward

        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        image_class_tokens = torch.zeros_like(x)

        # ============================================
        # Encoder

        if self.encoder.pos_embedding is not None:
            x = x + self.encoder.pos_embedding[:, 1:]
            class_token = class_token + self.encoder.pos_embedding[:, :1]
            image_class_tokens = image_class_tokens + class_token / num_patches

        for idx, module in enumerate(self.encoder.layers):
            for jdx, m in enumerate(module):
                x, class_token, other_outputs = m(
                    input=x,
                    class_token=class_token,
                    get_additional_outputs=True
                )

                if jdx != 0:
                    v = other_outputs['vectors'].squeeze(dim=1)  # size B,S',C
                    _, S, C = v.shape
                    s = int(math.sqrt(S))
                    v = v.permute(0, 2, 1).view(B, C, s, s)  # squarify
                    v = F.interpolate(v, mode='nearest', scale_factor=2 ** idx) / (2 ** (2 * idx))  # increase to original patch size
                    v = v.reshape(B, C, -1).permute(0, 2, 1)
                    v = v + (m.self_attention.out_proj.bias.view(1, 1, -1) / num_patches)  # sum bias
                    image_class_tokens = image_class_tokens + v
                else:
                    image_class_tokens = m.linear(image_class_tokens)

        # manual layernorm the image classification tokens
        if self.encoder.ln is not None:
            weight = self.encoder.ln.weight.view(1, 1, -1)  # [768] -> [1,1,768]
            bias = self.encoder.ln.bias.view(1, 1, -1) / num_patches  # [768] -> [1,1,768]
            cls_tokens = image_class_tokens.sum(dim=1, keepdim=True)  # [B,Seq Lenght,feat_dim] -> [B,1,768]

            # compute stats
            mean = cls_tokens.mean(dim=2, keepdim=True) / num_patches  # [B,1,768] -> [B,1,1]
            var = cls_tokens.var(dim=2, correction=0, keepdim=True)  # [B,1,768] -> [B,1,1]
            image_class_tokens = (image_class_tokens - mean) / (var + self.encoder.ln.eps).sqrt()  # [B,Seq Lenght,feat_dim] -> [B,Seq Lenght,feat_dim]
            image_class_tokens = image_class_tokens * weight + bias

        logits = self.heads(image_class_tokens.sum(dim=1))
        cam = F.linear(image_class_tokens, self.heads[0].weight, bias=None)
        cam = cam.permute((0, 2, 1))
        cam = cam.view(B, num_classes, image_size // patch_size, image_size // patch_size)

        if after_softmax:
            cam = F.softmax(cam, dim=1)
        if interpolation_mode is None:
            return logits, cam
        return logits, F.interpolate(cam, size=(image_size, image_size), mode=interpolation_mode)

    def forward_cam_layer(self, x, after_softmax=False):

        # include here layer dimension increase

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = sum(len(module) for module in self.encoder.layers) - len(self.encoder.layers)  # remove the pooling layers from the counts
        
        # =====================================
        # Virtually, manual forward

        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        # layer_class_tokens = torch.zeros(B, num_layers, x.size(2), device=x.device)
        layer_class_tokens = self.class_token.expand(x.shape[0], -1, -1)  # divide here for later 

        # ============================================
        # Encoder

        if self.encoder.pos_embedding is not None:
            x = x + self.encoder.pos_embedding[:, 1:]
            class_token = class_token + self.encoder.pos_embedding[:, :1]
            layer_class_tokens = layer_class_tokens + self.encoder.pos_embedding[:, :1]

        for idx, module in enumerate(self.encoder.layers):
            for jdx, m in enumerate(module):
                x, class_token, other_outputs = m(
                    input=x,
                    class_token=class_token,
                    get_additional_outputs=True
                )

                if jdx != 0:
                    v = other_outputs['vectors'].squeeze(dim=1)  # size B,S',C
                    v = v.sum(dim=1, keepdim=True) + m.self_attention.out_proj.bias.view(1, 1, -1)
                    layer_class_tokens = torch.cat([layer_class_tokens, v], dim=1)  # concat on seq dimension
                else:
                    layer_class_tokens = m.linear(layer_class_tokens)

        # manual layernorm the image classification tokens
        if self.encoder.ln is not None:
            weight = self.encoder.ln.weight.view(1, 1, -1)  # [768] -> [1,1,768]
            bias = self.encoder.ln.bias.view(1, 1, -1) / num_patches  # [768] -> [1,1,768]
            cls_tokens = layer_class_tokens.sum(dim=1, keepdim=True)  # [B,Seq Lenght,feat_dim] -> [B,1,768]

            # compute stats
            mean = cls_tokens.mean(dim=2, keepdim=True) / num_patches  # [B,1,768] -> [B,1,1]
            var = cls_tokens.var(dim=2, correction=0, keepdim=True)  # [B,1,768] -> [B,1,1]
            layer_class_tokens = (layer_class_tokens - mean) / (var + self.encoder.ln.eps).sqrt()  # [B,Seq Lenght,feat_dim] -> [B,Seq Lenght,feat_dim]
            layer_class_tokens = layer_class_tokens * weight + bias

        layer_class_tokens = layer_class_tokens[:, 1:] + layer_class_tokens[:, :1] / num_layers

        logits = self.heads(layer_class_tokens.sum(dim=1))
        cam = F.linear(layer_class_tokens, self.heads[0].weight, bias=None)
        cam = cam.permute((0, 2, 1))

        if after_softmax:
            cam = F.softmax(cam, dim=1)
        return logits, cam

    def forward_without_layers(self, x, layers_to_remove=[]):

         # include here layer dimension increase

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = sum(len(module) for module in self.encoder.layers) - len(self.encoder.layers)  # remove the pooling layers from the counts
        
        # =====================================
        # Virtually, manual forward

        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        # layer_class_tokens = torch.zeros(B, num_layers, x.size(2), device=x.device)
        layer_class_tokens = self.class_token.expand(x.shape[0], -1, -1)  # divide here for later 

        # ============================================
        # Encoder

        if self.encoder.pos_embedding is not None:
            x = x + self.encoder.pos_embedding[:, 1:]
            class_token = class_token + self.encoder.pos_embedding[:, :1]
            layer_class_tokens = layer_class_tokens + self.encoder.pos_embedding[:, :1]

        for idx, module in enumerate(self.encoder.layers):
            for jdx, m in enumerate(module):
                x, class_token, other_outputs = m(
                    input=x,
                    class_token=class_token,
                    get_additional_outputs=True
                )

                if jdx != 0:
                    v = other_outputs['vectors'].squeeze(dim=1)  # size B,S',C
                    v = v.sum(dim=1, keepdim=True) + m.self_attention.out_proj.bias.view(1, 1, -1)
                    layer_class_tokens = torch.cat([layer_class_tokens, v], dim=1)  # concat on seq dimension
                else:
                    layer_class_tokens = m.linear(layer_class_tokens)

        # manual layernorm the image classification tokens
        if self.encoder.ln is not None:
            weight = self.encoder.ln.weight.view(1, 1, -1)  # [768] -> [1,1,768]
            bias = self.encoder.ln.bias.view(1, 1, -1) / num_patches  # [768] -> [1,1,768]
            cls_tokens = layer_class_tokens.sum(dim=1, keepdim=True)  # [B,Seq Lenght,feat_dim] -> [B,1,768]

            # compute stats
            mean = cls_tokens.mean(dim=2, keepdim=True) / num_patches  # [B,1,768] -> [B,1,1]
            var = cls_tokens.var(dim=2, correction=0, keepdim=True)  # [B,1,768] -> [B,1,1]
            layer_class_tokens = (layer_class_tokens - mean) / (var + self.encoder.ln.eps).sqrt()  # [B,Seq Lenght,feat_dim] -> [B,Seq Lenght,feat_dim]
            layer_class_tokens = layer_class_tokens * weight + bias

        layers_to_keep = [0] + [a + 1 for a in range(num_layers) if a not in layers_to_remove]  # +1 since the first is the x[0]_0
        layer_class_tokens = layer_class_tokens[:, layers_to_keep]
        layer_class_tokens = layer_class_tokens.sum(dim=1)

        if self.encoder.ln is not None:
            layer_class_tokens = self.encoder.ln(layer_class_tokens)
        logits = self.heads(layer_class_tokens)
        return logits



    def forward_rollout(self, x, interpolation_mode=None):

        # include here layer dimension increase

        B = x.size(0)
        num_classes = self.num_classes
        patch_size = self.patch_size
        image_size = self.image_size
        num_patches = (image_size // patch_size) ** 2
        # =====================================
        # Virtually, manual forward

        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        image_class_tokens = 0  # should automatically broadcast to the appropiate shape

        # ============================================
        # Encoder

        if self.encoder.pos_embedding is not None:
            x = x + self.encoder.pos_embedding[:, 1:]
            class_token = class_token + self.encoder.pos_embedding[:, :1]

        for idx, module in enumerate(self.encoder.layers):
            for jdx, m in enumerate(module):
                x, class_token, other_outputs = m(
                    input=x,
                    class_token=class_token,
                    get_additional_outputs=True
                )

                if jdx != 0:
                    v = other_outputs['attn_map']  # size B,1,S'
                    _, _, S = v.shape
                    s = int(math.sqrt(S))
                    v = v.view(B, 1, s, s) 
                    v = F.interpolate(v, mode='nearest', scale_factor=2 ** idx) / (2 ** (2 * idx))  # increase to original patch size
                    image_class_tokens = image_class_tokens + v[:, 0]

        # manual layernorm the image classification tokens
        class_token = self.encoder.ln(class_token.squeeze(dim=1))
        logits = self.heads(class_token)

        if interpolation_mode is None:
            return logits, image_class_tokens
        return logits, F.interpolate(image_class_tokens, size=(image_size, image_size), mode=interpolation_mode)