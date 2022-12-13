"""
Model adapted from https://github.com/SatyamGaba/visual_question_answering
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ImageQuestionEncoder(nn.Module):

    def __init__(self, embed_size):
        super(ImageQuestionEncoder, self).__init__()
        vggnet_feat = models.vgg19(pretrained=True).features
        modules = list(vggnet_feat.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, embed_size),
                                nn.Tanh())    

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.cnn(image)                           
        img_feature = img_feature.view(-1, 512, 196).transpose(1,2) 
        img_feature = self.fc(img_feature)                         

        return img_feature


class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        """Stacked attention Module
        """
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        """Extract feature vector from image vector.
        """
        hi = self.ff_image(vi)
        hq = self.ff_questions(vq).unsqueeze(dim=1)
        ha = torch.tanh(hi+hq)
        if self.dropout:
            ha = self.dropout(ha)
        ha = self.ff_attention(ha)
        pi = torch.softmax(ha, dim=1)
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)
        u = vi_attended + vq
        return u

class QuestionEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QuestionEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     

    def forward(self, question):

        qst_vec = self.word2vec(question)                            
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                            
        _, (hidden, cell) = self.lstm(qst_vec)                       
        qst_feature = torch.cat((hidden, cell), 2)                  
        qst_feature = qst_feature.transpose(0, 1)                    
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                           

        return qst_feature

class QuestionAnswerModel(nn.Module):
    # num_attention_layer and num_mlp_layer not implemented yet
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size): 
        super(QuestionAnswerModel, self).__init__()
        self.num_attention_layer = 2
        self.num_mlp_layer = 1
        self.img_encoder = ImageQuestionEncoder(embed_size)
        self.qst_encoder = QuestionEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.san = nn.ModuleList([Attention(512, embed_size)]*self.num_attention_layer)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(embed_size, ans_vocab_size))
        self.attn_features = []  

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     
        qst_feature = self.qst_encoder(qst)                    
        vi = img_feature
        u = qst_feature
        for attn_layer in self.san:
            u = attn_layer(vi, u)
            
        combined_feature = self.mlp(u)
        return combined_feature