#!/usr/bin/env python
import os,sys
import numpy as np

import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable

def accu(pred,val,batch_l):

    correct=0
    total=0
    cor_seq=0
    for i in range(0,batch_l.shape[0]):
        mm=(pred[i,0:batch_l[i]].cpu().data.numpy() == val[i,0:batch_l[i]].cpu().data.numpy())
        correct+=mm.sum()
        total+=batch_l[i].sum()
        cor_seq+=mm.all()
    acc=correct/float(total)
    acc2=cor_seq/batch_l.shape[0]
    return acc,acc2

def vec_to_char(out_num):
    stri=""
    for cha in out_num:
        stri+=char_list[cha]
    return stri

def cal_prec_rec(Ypred,Ydata,conf):

    small=0.0000000001
    Ypred0=Ypred.cpu().data.numpy()
    Ydata0=Ydata.cpu().data.numpy()
    Ypred00=Ypred0>conf
    mm=Ypred00*Ydata0
    TP=mm.sum()
    A=Ydata0.sum()
    P=Ypred00.sum()
    precision=(TP+small)/(P+small)
    recall=(TP+small)/A

    return precision, recall

class Encoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Encoder,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)
        self.encoder_rnn = nn.LSTM(input_size=self.Nfea,hidden_size=self.hidden_dim,
                num_layers=self.NLSTM_layer,bias=True,
                batch_first=True,bidirectional=False)

        for param in self.encoder_rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self,X0,L0):

        batch_size=X0.shape[0]
        device=X0.device
        enc_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        enc_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X0)
        out,(encoder_hn,encoder_cn)=self.encoder_rnn(X,(enc_h0,enc_c0))
        last_step_index_list = (L0 - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        Z=out.gather(1,last_step_index_list).squeeze()
#        Z=torch.sigmoid(Z)
        Z=F.normalize(Z,p=2,dim=1)

        return Z

class Decoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Decoder,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)

#        self.decoder_rnn = nn.LSTM(input_size=self.Nfea,
        self.decoder_rnn = nn.LSTM(input_size=self.Nfea+self.hidden_dim,
            hidden_size=self.hidden_dim, num_layers=self.NLSTM_layer,
            bias=True, batch_first=True,bidirectional=False)

        for param in self.decoder_rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        self.decoder_fc1=nn.Linear(self.hidden_dim,self.Nfea)
        nn.init.xavier_normal_(self.decoder_fc1.weight.data)
        nn.init.normal_(self.decoder_fc1.bias.data)

    def forward(self, Z, X0, L0):

        batch_size=Z.shape[0]
        device=Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X0)
        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,self.Nseq,self.hidden_dim)
        ZX=torch.cat((Zm,X),2)

#        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(X0,(Z.view(1,-1,self.hidden_dim),dec_c0))
        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(ZX,(dec_h0,dec_c0))
        dec=self.decoder_fc1(dec_out)
        return dec

    def decoding(self, Z):
        batch_size=Z.shape[0]
        device=Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        seq=torch.zeros([batch_size,1],dtype=torch.long).to(device)
        seq[:,0]=self.Nfea-2

#        Xdata_onehot=torch.zeros([batch_size,1,self.Nfea],dtype=torch.float32).to(device)
#        Xdata_onehot[:,0,self.Nfea-2]=1
        Y = seq
        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,1,self.hidden_dim)

        decoder_hn=dec_h0
        decoder_cn=dec_c0
#        seq2=Xdata_onehot
        for i in range(self.Nseq):
            dec_h0=decoder_hn
            dec_c0=decoder_cn

            X = self.embedd(Y)
            ZX=torch.cat((Zm,X),2)
            dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(ZX,(dec_h0,dec_c0))
            dec=self.decoder_fc1(dec_out)
            Y= torch.argmax(dec,dim=2)
#            Xdata_onehot=torch.zeros([batch_size,self.Nfea],dtype=torch.float32).to(device)
#            Xdata_onehot=Xdata_onehot.scatter_(1,Y,1).view(-1,1,self.Nfea)
            seq=torch.cat((seq,Y),dim=1)
#            seq2=torch.cat((seq2,dec),dim=1)

        return seq #, seq2[:,1:]


# 把generator改为pgd
class Generator(nn.Module):
    def __init__(self, para, Pred, bias=True, ):
        super(Generator,self).__init__()

        self.seed_dim=para['seed_dim']
        self.hidden_dim=para['hidden_dim']
        self.model = Pred
        self.nb_iter = 40
        self.clip_min, self.clip_max = 0.0, 1.0
        self.eps, self.iter_eps = 0.3, 0.01
    
    # 均用第一维的
    def forward(self,Z, labels):
        adv_x=self.attack(Z,labels)
        return adv_x

    def sigle_step_attack(self,x,pertubation,labels):
        adv_x=x+pertubation
        # get the gradient of x
        adv_x=Variable(adv_x)
        adv_x.requires_grad = True
        loss_func=nn.CrossEntropyLoss()
        preds, _=self.model(adv_x)
        #print(type(preds), type(labels))
        loss=loss_func(preds,labels.long().cuda())
        
        self.model.zero_grad()
        loss.backward()
        grad=adv_x.grad.data
        #get the pertubation of an iter_eps
        pertubation=self.iter_eps * torch.abs(grad)
        adv_x = adv_x + pertubation
        
        pertubation=torch.clamp(adv_x,self.clip_min,self.clip_max)-x

        return pertubation

    def attack(self,x,labels):
        x_tmp=x+torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        pertubation=torch.zeros(x.shape).type_as(x).cuda()
        for i in range(self.nb_iter):
            pertubation=self.sigle_step_attack(x_tmp,pertubation=pertubation,labels=labels)
        
        adv_x = x+pertubation
        adv_x = torch.clamp(adv_x,self.clip_min,self.clip_max)
        return adv_x


class Critic(nn.Module):
    def __init__(self,para,bias=True):
        super(Critic,self).__init__()

        self.hidden_dim=para['hidden_dim']

        self.critic_fc1=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc1.weight.data)
        nn.init.normal_(self.critic_fc1.bias.data)

        self.critic_fc2=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc2.weight.data)
        nn.init.normal_(self.critic_fc2.bias.data)

        self.critic_fc3=nn.Linear(self.hidden_dim,1)
        nn.init.xavier_normal_(self.critic_fc3.weight.data)
        nn.init.normal_(self.critic_fc3.bias.data)

    def forward(self,Z0):

        D1=self.critic_fc1(Z0)
        D1=torch.relu(D1)
        D2=self.critic_fc2(D1)
        D2=torch.relu(D2)
        Dout=self.critic_fc3(D2)

        return Dout

    def clip(self,epsi=0.01):
        torch.clamp_(self.critic_fc1.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc1.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc2.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc2.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc3.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc3.bias.data,min=-epsi,max=epsi)


class Net(nn.Module):

    def __init__(self,para,bias=True):
        super(Net,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.Enc=Encoder(para)
        self.Dec=Decoder(para)
        
        self.Cri=Critic(para)
        self.Pred=Predictor(para)
        self.Gen=Generator(para, self.Pred)


    def AE(self, X0, L0, noise):

        Z = self.Enc(X0, L0)
#        print(Z.shape, noise.shape)
        Zn = Z+noise
        decoded = self.Dec(Zn, X0, L0)

        return decoded



class Predictor(nn.Module):
    def __init__(self,para,bias=True):
        super(Predictor,self).__init__()
        self.hidden_dim=para['hidden_dim']
        self.fc1 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)


    def forward(self, Z): #Z encoder之后的embedding
        y1, y2 = self.fc1(Z), self.fc2(Z)
        return y1, y2

    def Loss(self, y_real, y_pred):
        x = y_real - y_pred
        x = x**2
        return torch.sum(x) / y_real.shape[0]





def main():

    print("main")

if __name__=="__main__":
    main()





