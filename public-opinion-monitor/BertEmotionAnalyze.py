import os

from transformers import *
import torch.nn as nn
from torch.nn import *

import torch
import time

import argparse

class NewsData(torch.utils.data.Dataset):
    def __init__(self, root, is_train=1):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.data_num = 7346
        self.x_list = []
        self.y_list = []
        self.posi = []

        with open(root + '/Train_DataSet.csv', encoding='UTF-8') as f:
            for i in range(self.data_num + 1):
                line = f.readline()[:-1] + '这是一个中性的数据'

                data_one_str = line.split(',')[len(line.split(',')) - 2]
                data_two_str = line.split(',')[len(line.split(',')) - 1]

                if len(data_one_str) < 6:
                    z = len(data_one_str)
                    data_one_str = data_one_str + '，' + data_two_str[0:min(200, len(data_two_str))]
                else:
                    data_one_str = data_one_str
                if i == 0:
                    continue

                word_l = self.tokenizer.encode(data_one_str, add_special_tokens=False)
                if len(word_l) < 100:
                    while (len(word_l) != 100):
                        word_l.append(0)
                else:
                    word_l = word_l[0:100]

                word_l.append(102)
                l = word_l
                word_l = [101]
                word_l.extend(l)

                self.x_list.append(torch.tensor(word_l))

                self.posi.append(torch.tensor([i for i in range(102)]))

        with open(root + '/Train_DataSet_Label.csv', encoding='UTF-8') as f:
            for i in range(self.data_num + 1):
                # print(i)
                label_one = f.readline()[-2]
                if i == 0:
                    continue
                label_one = int(label_one)
                self.y_list.append(torch.tensor(label_one))

        # 训练集或者是测试集
        if is_train == 1:
            self.x_list = self.x_list[0:6000]
            self.y_list = self.y_list[0:6000]
            self.posi = self.posi[0:6000]
        else:
            self.x_list = self.x_list[6000:]
            self.y_list = self.y_list[6000:]
            self.posi = self.posi[6000:]

        self.len = len(self.x_list)

    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index], self.posi[index]

    def __len__(self):
        return self.len


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=10,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=3,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=3,help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    #类方法.from_pretrained()获取预训练模型,num_labels是分类的类数
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=opt.num_labels)
    return model

def get_data(opt):
    #NewsData继承于pytorch的Dataset类
    trainset = NewsData(opt.data_path,is_train = 1)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    testset = NewsData(opt.data_path,is_train = 0)
    testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    return trainloader,testloader


def train(epoch, model, trainloader, testloader, optimizer, opt):
    print('\ntrain-Epoch: %d' % (epoch + 1))
    model.train()
    start_time = time.time()
    print_step = int(len(trainloader) / 10)
    for batch_idx, (sue, label, posi) in enumerate(trainloader):
        if opt.gpu:
            sue = sue.cuda()
            posi = posi.cuda()
            label = label.unsqueeze(1).cuda()

        optimizer.zero_grad()
        # 输入参数为词列表、位置列表、标签
        outputs = model(sue, position_ids=posi, labels=label)

        loss, logits = outputs[0], outputs[1]
        loss.backward()
        optimizer.step()

        if batch_idx % print_step == 0:
            print("Epoch:%d [%d|%d] loss:%f" % (epoch + 1, batch_idx, len(trainloader), loss.mean()))
    print("time:%.3f" % (time.time() - start_time))


# def testFunc(epoch, model, trainloader, testloader, opt):
#     print('\ntest-Epoch: %d' % (epoch + 1))
#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (sue, label, posi) in enumerate(testloader):
#             if opt.gpu:
#                 sue = sue.cuda()
#                 posi = posi.cuda()
#                 labels = label.unsqueeze(1).cuda()
#                 label = label.cuda()
#             else:
#                 labels = label.unsqueeze(1)
#
#             outputs = model(sue, labels=labels)
#             loss, logits = outputs[:2]
#             _, predicted = torch.max(logits.data, 1)
#
#             total += sue.size(0)
#             correct += predicted.data.eq(label.data).cpu().sum()
#
#     s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
#     print(s)


if __name__ == '__main__':
    opt = get_train_args()
    model = get_model(opt)
    trainloader, testloader = get_data(opt)

    if opt.gpu:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    if not os.path.exists('./model.pth'):
        for epoch in range(opt.nepoch):
            train(epoch, model, trainloader, testloader, optimizer, opt)
            testFunc(epoch, model, trainloader, testloader, opt)
        torch.save(model.state_dict(), './model.pth')
    else:
        model.load_state_dict(torch.load('model.pth'))
        print('模型存在,直接test')
        testFunc(0, model, trainloader, testloader, opt)

