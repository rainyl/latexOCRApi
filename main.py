'''
Description: main
Author: Rainyl
Date: 2022-03-02 11:48:41
LastEditTime: 2022-03-08 18:13:12
'''
import base64

from fastapi import FastAPI, Form
from PIL import Image
from io import BytesIO

from infer import Inferer
from utils import verifyUser
from config import LATEX_OCR_TEMPLATE

app = FastAPI()
inferer = Inferer()

@app.get("/")
async def root():
    return {"message": "Welcome to use LaTexOCR by Rainyl Based on https://github.com/lukas-blecher/LaTeX-OCR !"}


@app.post("/api/v1/latexocr")
async def predict(img: str = Form(...), user: str = Form(...), token: str = Form(...)):
    # print(img)
    res = LATEX_OCR_TEMPLATE
    if not verifyUser(user, token):
        res["status"] = "error"
        res["data"] = "Invalid user or token"
        return res
    image = Image.open(BytesIO(base64.b64decode(img)))
    image.save('temp.png')
    pred = inferer.infer(image)
    return pred


@app.get("/api/v1/latexocr/test")
async def test():
    imgpath = "dataset/testimg/0000000.png"
    with open(imgpath, "rb") as f:
        imgstr = base64.b64encode(f.read())
    img = Image.open(BytesIO(base64.b64decode(imgstr)))
    r = inferer.infer(img)
    return r

