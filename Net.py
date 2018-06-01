import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from Data import *
from torch.autograd import Variable

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LSTMTagger(nn.Module):
    '''https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch'''

    def __init__(self, input_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)

        # self.embedding_reg = nn.Embedding(12, 32)
        # self.embedding_device = nn.Embedding(3592, hidden_dim - 8)

        self.hidden = self.init_hidden()

        self.trans_layer_1 = nn.Linear(12, 2 * hidden_dim)
        self.trans_layer_2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.trans_layer_output = nn.Linear(2 * hidden_dim, hidden_dim)

        self.liner_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.liner_2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.liner_3 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.liner_4 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        # self.liner_5 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
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
            true_val.append(data[0][1])
            # for i in range(start_time+1, end_time+1):
            # print(type(data[i]))
            data_temp = data[start_time + 1:end_time + 1][:, 1:]
            '''
                self.hidden = (torch.cat((self.embedding_reg(data[1][0].long().unsqueeze(0).cuda())[0][0],
                                      self.embedding_device(data[1][1].long().unsqueeze(0).cuda())[0][0])).unsqueeze(
                0).unsqueeze(0), Variable(
                torch.zeros(1, 1, self.hidden_dim)).cuda())

            self.hidden = (

            self.hidden = (torch.cat((self.embedding_reg(data[1][0].long().unsqueeze(0).cuda())[0][0],
                                      Variable(torch.zeros(24).float().cuda()))).unsqueeze(0).unsqueeze(0),
                           Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
            self,hidden = (1,Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
            '''
            reg_type = np.zeros(12)
            reg_type[int(data[1][0])] = 1
            reg_type = Variable(torch.Tensor(reg_type)).cuda()
            reg_type = F.relu(self.trans_layer_1(reg_type))
            reg_type = F.relu(self.trans_layer_2(reg_type))
            reg_type = F.relu(self.trans_layer_output(reg_type))
            self.hidden = (reg_type.unsqueeze(0).unsqueeze(0),
                               Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
            # print(type(data_temp.unsqueeze(0)),data_temp.unsqueeze(0).shape,data_temp.unsqueeze(0))

            # out, self.hidden = self.lstm(F.tanh(data_temp.unsqueeze(0)), self.hidden)
            out, self.hidden = self.lstm(data_temp.unsqueeze(0), self.hidden)

            # print(self.hidden)
            x = F.relu(self.liner_1(self.hidden[0][0]))
            x = F.relu(self.liner_2(x))
            x = F.relu(self.liner_3(x))
            x = F.relu(self.liner_4(x))
            # x = F.relu(self.liner_5(x))
            x = F.sigmoid(self.ouput_layer(x))
            predict.append(x[0])
        return torch.cat(predict), torch.cat(true_val)


def train():
    lstm = LSTMTagger(11, 32).cuda()
    # transed = TransData(reload=True)
    data = []
    # data = Data(reload=True)
    transed = TransData(data=data, reload=True, delete_data=True)

    train_sample = TrainRandomSampler(transed)
    valid_sample = ValidRandomSampler(transed)
    train_loader = DataLoader(
        dataset=transed,
        batch_size=60,  # 批大小
        num_workers=5,  # 多线程读取数据的线程数
        sampler=train_sample
    )
    valid_loader = DataLoader(
        dataset=transed,
        batch_size=60,  # 批大小
        num_workers=5,  # 多线程读取数据的线程数
        sampler=valid_sample
    )
    loss_function = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(lstm.parameters(), lr=0.1)
    lstm.load_state_dict(torch.load('./saved_model/lstm_layer4 0.792809.pkl'))



    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        total_num = 0
        total_correct_predict_num = 0
        total_activity_num = 0
        total_predict_num = 0

        for i, data in enumerate(train_loader):


            #print(i)

            lstm.zero_grad()
            # lstm.hidden = lstm.init_hidden()
            data = Variable(data.float()).cuda()
            predict, true_val = lstm(data)
            loss = loss_function(predict, true_val)
            total_loss += float(loss)
            #loss.backward()
            #optimizer.step()

            predict_num = torch.sum(predict > 0.5).float()
            activity_num = torch.sum(true_val).float()
            correct_predict_num = torch.sum((predict > 0.5).float() * true_val.float()).float()
            # pre = correct_predict_num / predict_num
            # recall = correct_predict_num / activity_num
            # f1_score = 2 * pre * recall / (pre + recall+0.00001) #加一个数防止nan
            # print(f1_score)
            # loss = 1 - f1_score
            # loss.backward()

            total_correct_predict_num += int(correct_predict_num)
            total_predict_num += int(predict_num)
            total_activity_num += int(activity_num)
            '''
            if int(pdt > 0.5):
                total_predict_num += 1
                if int(true == 1):
                    correct_predict_num += 1
            if int(true==1):
                total_activity_num += 1
'''
            total_num += 1

        try:
            pre = total_correct_predict_num / total_predict_num
        except:
            pre = 0

        recall = total_correct_predict_num / total_activity_num
        f1_score = 2 * pre * recall / (pre + recall + 0.0001)
        print('epoch:', epoch + 1)
        print('train avg loss:', total_loss / total_num)
        print('train precison:', pre, 'recall', recall)
        print('train F1 score:', f1_score)
        #torch.save(lstm.state_dict(), './saved_model/lstm_layer4 %.6f.pkl' % f1_score)

        # See what the scores are after training

        # with torch.no_grad():
        if 1:
            total_loss = 0
            total_num = 0
            correct_predict_num = 0
            total_activity_num = 0
            total_predict_num = 0
            total_correct_predict_num = 0

            for i, data in enumerate(valid_loader):
                lstm.zero_grad()
                # lstm.hidden = lstm.init_hidden()
                data = Variable(data.float()).cuda()
                predict, true_val = lstm(data)
                loss = loss_function(predict, true_val)
                total_loss += float(loss)

                predict_num = torch.sum(predict > 0.5).float()
                activity_num = torch.sum(true_val).float()
                correct_predict_num = torch.sum((predict > 0.5).float() * true_val.float()).float()

                total_correct_predict_num += int(correct_predict_num)
                total_predict_num += int(predict_num)
                total_activity_num += int(activity_num)
                total_num += 1

            try:
                pre = float(total_correct_predict_num / total_predict_num)
                recall = float(total_correct_predict_num / total_activity_num)
                f1_score = 2 * pre * recall / (pre + recall)
            except:
                pre = 0
                recall = 0
                f1_score = 0

            print('valid avg loss:', total_loss / total_num)
            print('valid precison:', pre, 'recall', recall)
            print('valid F1 score:', f1_score)
            print('')


def make_predict():
    lstm = LSTMTagger(11, 32).cuda()
    data = Data(reload=True)
    transed = TransData(data=data, reload=True, delete_data=False)
    data_loader = DataLoader(
        dataset=transed,
        batch_size=1,  # 批大小
        num_workers=1,  # 多线程读取数据的线程数
    )
    lstm.load_state_dict(torch.load('./saved_model/lstm_layer4 0.792809.pkl'))
    f = open('result.csv', 'w')
    for i, data in enumerate(data_loader):
        id = data[0][0][0]
        lstm.zero_grad()
        #lstm.hidden = lstm.init_hidden()
        data[0][0][3] = 31
        data = Variable(data.float()).cuda()
        predict, true_val = lstm(data)
        if int(predict[0] > 0.5):
            f.write(str(int(id)) + '\n')
    f.close()



if __name__ == '__main__':
    train()
    #make_predict()
