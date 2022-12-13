import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import glob
import os
import time
import numpy as np
import argparse
import argparse

from data.data_utilities import make_vocab_questions, make_vocab_answers, VocabDict
from data.prepareVQAMapping import prepare_vqa_dataset
from data.VQADataset import VisualQuestionAnswerDataset
from models.models import QuestionAnswerModel
from engine import train, test


# declare path for data
data_path = "./data"
zipped_path = os.path.join(data_path, "zip")
annotations_path = os.path.join(data_path, "annotations")
questions_path = os.path.join(data_path, "questions")
image_path = os.path.join(data_path, "images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=24, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='size of each batch to run')
    opt = parser.parse_args()

    EPOCHS = opt.epochs
    batch_size = opt.batch_size

    # prepare vocabulary from questions and answers
    make_vocab_questions(questions_path)
    make_vocab_answers(annotations_path, 1000)

    # valid answerset
    vocab_answer_file = annotations_path+'/vocab_answers.txt'
    answer_dict = VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list) 

    train_dataset = prepare_vqa_dataset(image_path, questions_path, annotations_path, valid_answer_set, data_type="train2014")
    val_dataset = prepare_vqa_dataset(image_path, questions_path, annotations_path, valid_answer_set, data_type="val2014")

    # image transformation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                        ])


    train_dataset_vqa = VisualQuestionAnswerDataset(train_dataset, questions_path, annotations_path, transform=train_transform, max_q_length=25)
    val_dataset_vqa = VisualQuestionAnswerDataset(val_dataset, questions_path, annotations_path, transform=test_transform, max_q_length=25)

    dataloader_args = dict(shuffle=True, batch_size=64, num_workers=2, pin_memory=True)

    vqa_train_loader = torch.utils.data.DataLoader(train_dataset_vqa, **dataloader_args)
    vqa_val_loader = torch.utils.data.DataLoader(val_dataset_vqa, **dataloader_args)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QuestionAnswerModel(embed_size=1024, qst_vocab_size=15336, ans_vocab_size=1000, word_embed_size=300,
            num_layers=2,
            hidden_size=512).to(device)

    optimizer = optim.Adam(model.parameters())

    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []

    for epoch in range(EPOCHS):
        print(f"Epoch run- {epoch+1}")
        train_acc, train_l = train(model, device, vqa_train_loader, optimizer, criterion)
        test_acc, test_l = test(model, device, vqa_val_loader, criterion)
        train_accuracy.append(train_acc)
        train_loss.append(train_l)
        test_accuracy.append(test_acc)
        test_loss.append(test_l)