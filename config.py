'''
Description: Config
Author: Rainyl
Date: 2022-03-02 16:19:09
LastEditTime: 2022-03-02 21:05:19
'''
import os
import json
from typing import Dict, Sequence

from munch import Munch
import albumentations as alb
from albumentations.pytorch import ToTensorV2

CONF_PATH = 'config.json'
VALID_TOKEN_PATH = 'validTokens.json'
LATEX_OCR_TEMPLATE = {"status": "-1", "data": ""}

if not os.path.exists(CONF_PATH):
    raise Exception('Config file not found!')

with open(CONF_PATH, 'r', encoding="utf-8") as f:
    CONFIG: Munch = Munch(json.load(f))


if not os.path.exists(VALID_TOKEN_PATH):
    raise Exception('Valid token file not found!')

with open(VALID_TOKEN_PATH, 'r', encoding="utf-8") as f:
    VALID_TOKENS: Dict[str, Sequence[str]] = json.load(f)

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)
