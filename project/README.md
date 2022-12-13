# Visual Question Answering


Visual Question Answering attempts to provide an answer in natural language given a natural language question and an image. The model tries to understand the context from the asked question and image while answering. This is useful in cases such as providing assistance to visually-impaired users, reporting on surveillance data.

This repo contains code to train a convolutional and natural language based model on VQA dataset.

## Highlights

## Install the dependencies

pip install -r requirements.txt


## Dataset

[VQA dataset](https://visualqa.org/) has been used for the model training and inference. To download and prepare the dataset-

```
python3 prepare_data.py './data'
```

## Results

## References

1. [Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf)

2. [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1908.07490.pdf)

3. [Visual Question Answering dataset](https://visualqa.org/)


