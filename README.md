<!--
 * @Description: LatexOCRApi
 * @Author: Rainyl
 * @Date: 2022-03-09 11:04:43
 * @LastEditTime: 2022-03-09 11:15:48
-->
# latexOCRApi

A Latex OCR API based on [LaTex-OCR](https://github.com/lukas-blecher/LaTeX-OCR) and [fastApi](https://github.com/tiangolo/fastapi)

## Install

1. create environment and install dependences

`pip install -r requirements.txt`

2. prepare `validTokens.json`

it has to be the following format:

`{
    "user" : [
        "token1",
        "token2",
        "token3",
        ...
    ]
}`

3. run `uvicorn main:app --reload`

## Deploy

You can deploy by following the FastApi's document, https://fastapi.tiangolo.com/deployment/

## License

GPLv3 for non-commercial use only

For commercial use, please contact me.
