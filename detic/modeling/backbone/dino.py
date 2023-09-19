# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torchvision.transforms.functional as Fv

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN

from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
from centernet.modeling.backbone.bifpn import BiFPN
# from .checkpoint import load_checkpoint
from IPython import embed


class Dinov2Backbone(Backbone):
    def __init__(self, name='dinov2_vits14', embed_dim=384, out_indices=[1]):
        super().__init__()
        self.embed = torch.hub.load('facebookresearch/dinov2', name)
        self.embed_dim = embed_dim
        self.patch_size = 14
        # XXX: something is wrong where out_indices != [1].
        self.out_indices = out_indices
        self._out_features = [f'dino{i}' for i in self.out_indices]
        self._out_feature_channels = {
            f'dino{i}': self.embed_dim * (2 ** (2*(i-1))) 
            for i in self.out_indices
        }
        self._out_feature_strides = {
            f'dino{i}': 2 ** (i+2) 
            for i in self.out_indices
        }
        print(self._out_features)
        print(self._out_feature_channels)
        print(self._out_feature_strides)
        self._freeze_stages()

    def _freeze_stages(self):
        self.embed.eval()
        for param in self.embed.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def forward_features(self, x, masks=None):
        x = self.embed.prepare_tokens_with_masks(x, masks)

        for blk in self.embed.blocks:
            x = blk(x)

        x_norm = self.embed.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x):
        # resize to a multiple of patch size
        _, _, h, w = x.shape
        p = self.patch_size
        s2 = (2 ** (max(self.out_indices)-1))
        hh = int(h/p/s2) * s2
        ww = int(w/p/s2) * s2
        x = Fv.resize(x, (hh * p, ww * p), antialias=True)

        d = self.embed.forward_features(x)
        x = d['x_norm_patchtokens']
        out = {}
        for i, k in zip(self.out_indices, self._out_features):
            j = 2 ** (i-1)
            # xi = x[:, :, ::j].reshape(
            #     x.shape[0], self.embed_dim * j * 2, 
            #     int(hh/j), int(ww/j))

            hi = int(hh/j)
            wi = int(ww/j)
            xi = x.view(-1, hi, wi, self._out_feature_channels[k]).permute(0, 3, 1, 2).contiguous()
            # print(k, xi.shape, (hi, wi, self._out_feature_channels[k]))
            # print(x.shape)
            out[k] = xi
        return out

# {'swin1': torch.Size([1, 192, 56, 56]),
#  'swin2': torch.Size([1, 384, 28, 28]),
#  'swin3': torch.Size([1, 768, 14, 14])}
# {'dino1': torch.Size([1, 384, 32, 32]),
#  'dino2': torch.Size([1, 768, 16, 16]),
#  'dino3': torch.Size([1, 1536, 8, 8])}


size2config = {
    'S': {
        'name': 'dinov2_vits14',
        'embed_dim': 384,
    },
    'B': {
        'name': 'dinov2_vitb14',
        'embed_dim': 768,
    },
    'L': {
        'name': 'dinov2_vitl14',
        'embed_dim': 1024,
    },
    'G': {
        'name': 'dinov2_vitg14',
        'embed_dim': 1536,
    },
}

@BACKBONE_REGISTRY.register()
def build_dino_backbone(cfg, input_shape):
    """
    """
    out_indices = cfg.MODEL.DINO.OUT_FEATURES
    config = size2config[cfg.MODEL.DINO.SIZE]
    model = Dinov2Backbone(out_indices=out_indices, **config)
    return model


@BACKBONE_REGISTRY.register()
def build_dino_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_dino_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_dino_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_dino_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    backbone = BiFPN(
        cfg=cfg,
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=cfg.MODEL.BIFPN.OUT_CHANNELS,
        norm=cfg.MODEL.BIFPN.NORM,
        num_levels=cfg.MODEL.BIFPN.NUM_LEVELS,
        num_bifpn=cfg.MODEL.BIFPN.NUM_BIFPN,
        separable_conv=cfg.MODEL.BIFPN.SEPARABLE_CONV,
    )
    return backbone