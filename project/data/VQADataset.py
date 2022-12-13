import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_utilities import VocabDict

class VisualQuestionAnswerDataset(Dataset):
    def __init__(self, data_set, questions_path, annotations_path, transform=None, max_q_length=30):
        self.data_set = data_set
        self.question_vocab = VocabDict(questions_path+'/vocab_questions.txt')
        self.answer_vocab = VocabDict(annotations_path+'/vocab_answers.txt')
        self.transform = transform
        self.max_q_length = max_q_length

    
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index):
        # one sample of data has following keys- 
        # question_id, image_id, image_name, question_string, question_token, answers_all, valid_answers
        query_img = cv2.imread(self.data_set[index]["image_name"])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        qst2idc = np.array([self.question_vocab.word2idx('<pad>')] * self.max_q_length)
        qst2idc[:len(self.data_set[index]['question_token'])] = [self.question_vocab.word2idx(w) for w in self.data_set[index]['question_token']]
        ans2idc = [self.answer_vocab.word2idx(w) for w in self.data_set[index]['valid_answers']]
        ans2idx = np.random.choice(ans2idc)

        sample = {'image': query_img, 'question': qst2idc, 'answer': ans2idx}

        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
        return sample        