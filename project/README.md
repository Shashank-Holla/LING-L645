# Visual Question Answering


Visual Question Answering attempts to provide an answer in natural language given a natural language question and an image. The model tries to understand the context from the asked question and image while answering. This is useful in cases such as providing assistance to visually-impaired users, reporting on surveillance data.

This repository explores convolution based model for image feature extraction and LSTM based model for natural language extractions on VQA dataset. The primary task explored is to accurately answer questions (close question-with single word answers).

## Highlights

1. Image context extraction model pretrained is VGG19, pretrained on ImageNet dataset.
2. LSTM based language model used for encoding natural language questions.
3. Stacked Attention Network for answering questions. 



## Install the dependencies

To install the dependencies, execute-
```
pip install -r requirements.txt
```


## Dataset

[VQA dataset](https://visualqa.org/) has been used for the model training and inference. This dataset contains open-ended questions about images. The images are from COCO dataset and contain about 80,000 training images.

| Datatype   | No of images | No. of questions | No. of answers |
|------------|--------------|------------------|----------------|
| Train      | 82,783       | 443,757          | 4,437,570      |
| Validation | 40,504       | 214,354          | 2,143,540      |


To download and prepare the dataset-

```
python3 prepare_data.py './data'
```

## Model

For the image feature extraction, pre-trained VGG19 model has been used. The last layers of the model has been replaced with fully connected layer to provide the context vector for the images. The initial layers of the image encoder model are frozen and only the last layer is trained.

2 layered LSTM model is considered for extracting context from the questions. Attention is then applied on the latent vectors from image and question encoder blocks.

## Training and inference

While training the model, the following arguments can be passed-

a. epochs - Number of epochs to train the model

b. batch_size - Number of images in a batch to train

```
python3 main.py --epochs 8 --batch_size 32

```

## References

1. [Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf)

2. [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1908.07490.pdf)

3. [Visual Question Answering dataset](https://visualqa.org/)

4. https://github.com/Cadene/vqa.pytorch

## TODO

1. Implement GradCAM to understand the focus of the model while trying to answer the question.

2. Explore multi-word answers

3. Explore language transformers to extract encoding from questions.
