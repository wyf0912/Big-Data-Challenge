import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from Data import *
from torch.autograd import Variable


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class LSTMTagger(nn.Module):
    '''https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch'''

    def __init__(self, input_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.hidden = self.init_hidden()
        self.liner_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.liner_2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.ouput_layer = nn.Linear(2 * hidden_dim, 1)

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, batch_data):
        predict = []
        true_val = []
        for data in batch_data:
            start_time = int(data[0][2])
            end_time = int(data[0][3])
            if int(end_time - start_time > 2):
                true_val.append(data[0][1])
                # for i in range(start_time+1, end_time+1):
                # print(type(data[i]))
                data_temp = data[start_time + 1:end_time + 1]  #
                # print(type(data_temp.unsqueeze(0)),data_temp.unsqueeze(0).shape,data_temp.unsqueeze(0))
                cell_data, self.hidden = self.lstm(data_temp.unsqueeze(0), self.hidden)
                # print(self.hidden)
                x = F.relu(self.liner_1(self.hidden[0][0]))
                x = F.relu(self.liner_2(x))
                x = F.sigmoid(self.ouput_layer(x))
                predict.append(x)
        return predict, true_val


def train():
    lstm = LSTMTagger(12, 6).cuda()
    transed = TransData(reload=True)
    train_sample = TrainRandomSampler(transed)
    train_loader = DataLoader(
        dataset=transed,
        batch_size=100,  # 批大小
        num_workers=5,  # 多线程读取数据的线程数
        sampler=train_sample
    )
    loss_function = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(lstm.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        total_num = 0
        correct_predict_num = 0
        total_activity_num = 0
        total_predict_num = 0

        for i, data in enumerate(train_loader):
            lstm.zero_grad()
            lstm.hidden = lstm.init_hidden()
            data = Variable(data.float()).cuda()
            predict, true_val = lstm(data)
            for i, pdt in enumerate(predict):
                '''这里batch个数暂时不固定'''
                true = true_val[i].unsqueeze(0)
                loss = loss_function(pdt, true)
                total_loss += float(loss)
                loss.backward(retain_graph=True)
                if int(pdt > 0.5):
                    total_predict_num += 1
                    if int(true == 1):
                        correct_predict_num += 1
                if int(true==1):
                    total_activity_num += 1

            total_num += len(predict)
            optimizer.step()

        print('avg loss:', total_loss / total_num)
        pre = correct_predict_num/total_predict_num
        recall =correct_predict_num/total_activity_num
        print('precison:',pre,'recall',recall)
        print('F1 score:',2*pre*recall/(pre+recall))

    # See what the scores are after training
    '''
        with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)
    '''
    #torch.save(G.state_dict(), 'params_new_new_noamount %.6f.pkl' % (auc))


if __name__ == '__main__':
    train()
