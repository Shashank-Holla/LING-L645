# Visual Question Answering


Visual Question Answering attempts to provide an answer in natural language given a natural language question and an image. The model tries to understand the context from the asked question and image while answering. This is useful in cases such as providing assistance to visually-impaired users, reporting on surveillance data.

This repo contains code to train a convolutional and natural language based model on VQA dataset.

## Highlights

1. Pytorch framework based Convolution - Language model trained on Visual Question Answering dataset.



## Install the dependencies

To install the dependencies, execute-
```
pip install -r requirements.txt
```


## Dataset

[VQA dataset](https://visualqa.org/) has been used for the model training and inference. To download and prepare the dataset-

```
python3 prepare_data.py './data'
```



## Training and inference

```
python3 main.py --epochs 8 --batch_size 32

a. epochs - Number of epochs to train the model

b. batch_size - Number of images in a batch to train
```

## References

1. [Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf)

2. [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1908.07490.pdf)

3. [Visual Question Answering dataset](https://visualqa.org/)

## TODO

1. Implement GradCAM to understand the focus of the model while trying to answer the question.

2. Explore multi-word answers
