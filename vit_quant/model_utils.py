import torch
import typing
# import utils
import timm
import os
import logging


def get_model(name, ):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    net = timm.create_model(name, pretrained=True)

    net.cuda()
    net.eval()
    return net



def stem_layer_forward(model, calib_data, dev):
    if isinstance(model, timm.models.VisionTransformer):
        model.patch_embed = model.patch_embed.to(dev)
        model.pos_drop = model.pos_drop.to(dev)

        inps = model.patch_embed(calib_data)
        inps = model._pos_embed(inps)
        inps = model.patch_drop(inps)
        inps = model.norm_pre(inps)

        dtype = next(iter(model.parameters())).dtype

        model.patch_embed = model.patch_embed.cpu()
        model.pos_drop = model.pos_drop.cpu()
        kwargs = {}
    elif isinstance(model, timm.models.Eva):
        model.patch_embed = model.patch_embed.to(dev)
        model._pos_embed = model._pos_embed.to(dev)

        inps = model.patch_embed(calib_data)
        inps, rot_pos_embed = model._pos_embed(inps)
        inps = model.norm_pre(inps)

        dtype = next(iter(model.parameters())).dtype

        model.patch_embed = model.patch_embed.cpu()
        model.pos_drop = model.pos_drop.cpu()

    return model, inps, kwargs