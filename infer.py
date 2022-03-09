'''
Description: Infer
Author: Rainyl
Date: 2022-03-02 11:50:26
LastEditTime: 2022-03-09 10:20:22
'''
from typing import Union
# from dataset.dataset import test_transform
# import cv2
# import pyperclip
from PIL import Image
import os
# import sys
# import argparse
import logging
# import re

import numpy as np
import torch
# from torchvision import transforms
# from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from config import CONFIG, test_transform, LATEX_OCR_TEMPLATE
# from dataset.latex2png import tex2pil
from models import get_model, Model
from utils import pad, post_process, token2str


def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def initialize(arguments=None):
    logging.getLogger().setLevel(logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # args = CONFIG.copy()
    print(CONFIG)
    model = get_model(CONFIG)
    model.load_state_dict(
        torch.load(CONFIG.checkpoint, map_location=CONFIG.device)
    )

    image_resizer: Union[None, torch.nn.Module] = None
    if not CONFIG.no_resize and os.path.exists(CONFIG.ckpt_resize):
        image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(CONFIG.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                 preact=True, stem_type='same', conv_layer=StdConv2dSame).to(CONFIG.device)
        image_resizer.load_state_dict(torch.load(CONFIG.ckpt_resize, map_location=CONFIG.device))
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=CONFIG.tokenizer)
    return model, image_resizer, tokenizer


def call_model(img: Image, model: Model, image_resizer: Model, tokenizer: PreTrainedTokenizerFast):
    encoder, decoder = model.encoder, model.decoder
    w, h = img.size
    # img = img.resize((w*3, h*3), Image.LANCZOS)
    img = minmax_size(pad(img), CONFIG.max_dimensions, CONFIG.min_dimensions)
    if image_resizer is not None and not CONFIG.no_resize:
        with torch.no_grad():
            input_image = img.convert('RGB').copy()
            r, w, h = 1, input_image.size[0], input_image.size[1]
            for _ in range(10):
                h = int(h * r)  # height to resize
                logging.info(f"{r}, {img.size} -> {(w, h)}")
                img = pad(
                    minmax_size(
                        input_image.resize(
                            (w, h), 
                            Image.BILINEAR if r > 1 else Image.LANCZOS
                        ), CONFIG.max_dimensions, CONFIG.min_dimensions))
                t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                w = (image_resizer(t.to(CONFIG.device)).argmax(-1).item()+1)*32
                if (w == img.size[0]):
                    break
                r = w / img.size[0]
    else:
        img = np.array(pad(img).convert('RGB'))
        t = test_transform(image=img)['image'][:1].unsqueeze(0)
    img.save("call_model_resized.png")
    im = t.to(CONFIG.device)

    with torch.no_grad():
        model.eval()
        encoded = encoder(im.to(CONFIG.device))
        dec = decoder.generate(
            torch.LongTensor([CONFIG.bos_token])[:, None].to(CONFIG.device),
            CONFIG.max_seq_len,
            eos_token=CONFIG.eos_token,
            context=encoded.detach(),
            temperature=CONFIG.get('temperature', .25)
        )
        pred = post_process(token2str(dec, tokenizer)[0])
    # No use for api
    # try:
    #     pyperclip.copy(pred)
    # except:
    #     pass
    return pred


# def output_prediction(pred, args):
#     print(pred, '\n')
#     if args.show or args.katex:
#         try:
#             if args.katex:
#                 raise ValueError
#             tex2pil([f'$${pred}$$'])[0].show()
#         except Exception as e:
#             # render using katex
#             import webbrowser
#             from urllib.parse import quote
#             url = 'https://katex.org/?data=' + \
#                 quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
# "strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
#             webbrowser.open(url)


class Inferer(object):
    def __init__(self):
        try:
            self.model, self.image_resizer, self.tokenizer = initialize()
        except Exception as e:
            raise RuntimeError(f'Failed to initialize model: {e}')

    def infer(self, img: Image):
        res = LATEX_OCR_TEMPLATE
        try:
            pred = call_model(img, self.model, self.image_resizer, self.tokenizer)
            pred = pred.replace('<', '\\lt ').replace('>', '\\gt ')
            res["status"] = "success"
            res["data"] = [pred]  # type: ignore
            print(res)
            return res
        except Exception as e:
            res["status"] = "error"
            res["data"] = str(e)
            return res

def inferTex(img: Image):
    model, image_resizer, tokenizer = initialize()
    pred = call_model(img, model, image_resizer, tokenizer)
    return pred


if __name__ == "__main__":
    from pathlib import Path
    testImgPath = Path('dataset/testimg')
    imgs = list(testImgPath.glob("*.png"))
    image = Image.open(imgs[0])
    r = inferTex(image)
    print(r)
