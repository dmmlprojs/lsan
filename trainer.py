from transformer import *
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from LSAN import *
from torch.autograd import Variable
from sklearn import metrics



class LSAN_trainer:
    def __init__(self, lsan_model:LSAN, train_dataloader, validate_dataloader, test_dataloader, with_cuda=0,lr=0.001,output_dir=None):
        super().__init__()
        self.device = torch.device("cuda:0" if with_cuda==1 else "cpu")
        self.lsan = lsan_model.to(self.device)
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader
        self.optim = Adam(self.lsan.parameters(), lr=lr)
        self.output_dir = output_dir
        self.BCELoss = nn.BCELoss()

    def evaluate(selfs, pred, label):
        correct_num = 0
        predict_num = 0
        ground_truth_num = 0
       
        pred_pos = (pred == 1)
        pred_neg = (pred == 0)
        label_pos = (label == 1)
        label_neg = (label == 0)

        TP = (pred_pos & label_pos)
        FP = (pred_pos & label_neg)
        FN = (pred_neg & label_pos)
        TN = (pred_neg & label_neg)
        
        TP, FP, FN, TN = torch.sum(TP.int()), torch.sum(FP.int()), torch.sum(FN.int()), torch.sum(TN.int())
        return TP, FP, FN, TN 

    def train(self, epoch):

        avg_loss = 0.0
        step = 0
        for data in self.train_dataloader:
            step += 1

            padding_input, input_labels, x, y = data
            padding_input, input_labels = Variable(padding_input), Variable(input_labels)
            padding_input, input_labels = padding_input.to(self.device), input_labels.to(self.device)
            predict_output = self.lsan(padding_input).squeeze(1)
            
            predict_loss = self.BCELoss(predict_output, input_labels.type(torch.FloatTensor).to(self.device))

            loss = predict_loss        

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            avg_loss = avg_loss + loss.item()
            print(loss.item(), avg_loss/step)
        print("Done for epoch {}".format(epoch))


    def validate(self,epoch):
        step = 0
        avg_loss = 0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        
        labels = np.array([])
        preds = np.array([])

        for data in self.validate_dataloader:

            step += 1

            padding_input, input_labels, x, y = data
            padding_input, input_labels = Variable(padding_input), Variable(input_labels)
            padding_input, input_labels = padding_input.to(self.device), input_labels.to(self.device)

            predict_output = self.lsan(padding_input).squeeze(1)

            predict_loss = self.BCELoss(predict_output, input_labels.type(torch.FloatTensor).to(self.device))
            
            labels_temp = input_labels.cpu().numpy()
            preds_temp  = predict_output.detach().cpu().numpy()
            labels = np.concatenate((labels, labels_temp))
            preds = np.concatenate((preds, preds_temp))


            loss = predict_loss      
            t = Variable(torch.Tensor([0.45])).to(self.device)

            predict_label = (predict_output > t).float().to(self.device)
            predict_label_int = predict_label.type(input_labels.dtype)
            temp_TP, temp_FP, temp_FN, temp_TN  = self.evaluate(predict_label_int, input_labels)
            total_TP += temp_TP
            total_FP += temp_FP
            total_FN += temp_FN
            total_TN += temp_TN

            avg_loss += loss.item()


        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        if((total_TP == 0) & (total_FP == 0) & (total_FN ==0)):
            precision, recall = 1, 1
        elif(((total_TP == 0) & (total_FP == 0) & (total_FN !=0)) | ((total_TP == 0) & (total_FP == 0) & (total_FN !=0))):
            precision, recall = 0, 0
        else:
            recall, precision = float(total_TP)/float(total_TP+total_FN), float(total_TP)/float(total_TP+total_FP)

        str = "Validation epoch:{} auc:{} precision:{} recall:{}".format(epoch, auc, precision, recall)
        print("Validation epoch:{} auc:{} precision:{} recall:{}".format(epoch, auc, precision, recall))
        f = open(self.output_dir,'a+')
        f.write(str+'\n')

        f.close()

    def test(self,epoch):
        step = 0
        avg_loss = 0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        
        labels = np.array([])
        preds = np.array([])

        for data in self.test_dataloader:

            step += 1

            padding_input, input_labels, x, y = data
            padding_input, input_labels = Variable(padding_input), Variable(input_labels)
            padding_input, input_labels = padding_input.to(self.device), input_labels.to(self.device)

            predict_output = self.lsan(padding_input).squeeze(1)

            predict_loss = self.BCELoss(predict_output, input_labels.type(torch.FloatTensor).to(self.device))
            
            labels_temp = input_labels.cpu().numpy()
            preds_temp  = predict_output.detach().cpu().numpy()
            labels = np.concatenate((labels, labels_temp))
            preds = np.concatenate((preds, preds_temp))


            loss = predict_loss      
            t = Variable(torch.Tensor([0.45])).to(self.device)
            predict_label = (predict_output > t).float().to(self.device)
            predict_label_int = predict_label.type(input_labels.dtype)
            temp_TP, temp_FP, temp_FN, temp_TN  = self.evaluate(predict_label_int, input_labels)
            total_TP += temp_TP
            total_FP += temp_FP
            total_FN += temp_FN
            total_TN += temp_TN

            avg_loss += loss.item()


        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        if((total_TP == 0) & (total_FP == 0) & (total_FN ==0)):
            precision, recall = 1, 1
        elif(((total_TP == 0) & (total_FP == 0) & (total_FN !=0)) | ((total_TP == 0) & (total_FP == 0) & (total_FN !=0))):
            precision, recall = 0, 0
        else:
            recall, precision = float(total_TP)/float(total_TP+total_FN), float(total_TP)/float(total_TP+total_FP)
        F1 = 2*(recall*precision)/(recall+precision)
        str = "Testing epoch:{} auc:{} precision:{} recall:{} F1:{}".format(epoch, auc, precision, recall, F1)
        print("Testing epoch:{} auc:{} precision:{} recall:{} F1:{}".format(epoch, auc, precision, recall, F1))
        f = open(self.output_dir,'a+')
        f.write(str+'\n')

        f.close()

