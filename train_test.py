import numpy as np
import torch.cuda
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from config import *


def LossPredLoss():
    pass


def accu(pred, val, batch_l):
    correct = 0
    total = 0
    cor_seq = 0
    for i in range(0, batch_l.shape[0]):
        try:
            mm = (pred[i, 0:batch_l[i]].cpu().data.numpy()
                  == val[i, 0:batch_l[i]].cpu().data.numpy())
            correct += mm.sum()
            total += batch_l[i].sum()
            cor_seq += mm.all()
        except:
            print(pred[i, 0:batch_l[i]].cpu().data.numpy(),
                  val[i, 0:batch_l[i]].cpu().data.numpy())
            return 0, 0

    acc = correct / float(total)
    acc2 = cor_seq / batch_l.shape[0]
    #    print(correct,total,acc,cor_seq)
    return acc, acc2


def vec_to_char(out_num):
    stri = ""
    for cha in out_num:
        stri += CHAR_LIST[cha]
    return stri


def cal_prec_rec(Ypred, Ydata, conf):
    small = 0.0000000001
    Ypred0 = Ypred.cpu().data.numpy()
    Ydata0 = Ydata.cpu().data.numpy()
    Ypred00 = Ypred0 > conf
    mm = Ypred00 * Ydata0
    TP = mm.sum()
    A = Ydata0.sum()
    P = Ypred00.sum()
    precision = (TP + small) / (P + small)
    recall = (TP + small) / A
    return precision, recall


def train_test(model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, device):
    print('>> Train a Model')
    model.to(device)
    mean0 = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    mean_seed = torch.zeros(BATCH_SIZE, SEED_DIM)
    total_train_step = 0
    total_test_step = 0
    train_acc, test_acc = [], []
    for epoch in range(epochs):
        std = STD0 * np.power(STD_DECAY_RATIO, epoch) + STD00
        model.train()
        correct = 0
        pred_list = []
        for i, data in enumerate(train_loader):
            batch_x, batch_l, batch_p = data
            batch_x = batch_x.to(device)
            batch_l = batch_l.to(device)
            batch_p = batch_p.to(device)
            batch_x2 = batch_x[:, 1:]
            optimizer.zero_grad()
            noise = torch.normal(mean=mean0, std=std).to(device)
            out_decoding = model.AE(batch_x, batch_l, noise)
            out2 = out_decoding[:, :-1]
            loss1 = criterion(out2.reshape(-1, NFEA), batch_x2.reshape(-1))
            Z_real = model.Enc(batch_x, batch_l)
            pred = model.BinC(Z_real)
            pred_p = torch.max(F.softmax(pred, dim=1), 1)[1]
            correct_prediction = np.equal(pred_p.cpu().numpy(), batch_p.cpu().numpy())
            correct += np.sum(correct_prediction)
            loss2 = model.BinC.Loss(batch_p, pred_p)
            loss = loss1 + loss2
            total_train_step += 1
            # if total_train_step % 120 == 0:
                # train_acc.append(accuracy_score(batch_p, pred_p.argmax(1)))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 15 != 14:
                continue
            _, out_num_AE = torch.max(out_decoding, 2)
            acc, acc2 = accu(out_num_AE, batch_x2, batch_l)
            print("reconstruction accuracy:", acc, acc2)
            for k in range(0, 2):
                out_string = vec_to_char(batch_x2[k])
                print("real: ", out_string)
                out_string = vec_to_char(out_num_AE[k])
                print("AE  : ", out_string)
        print(correct, len(train_loader) * BATCH_SIZE)
        acc = correct / (len(train_loader) * BATCH_SIZE)
        print(f"Train loss: {loss}, Train acc: {acc}")
        # test
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch_x, batch_l, batch_p = data
                batch_x2 = batch_x[:, 1:]
                batch_x = batch_x.to(device)
                batch_l = batch_l.to(device)
                batch_p = batch_p.to(device)
                batch_x2 = batch_x[:, 1:]
                noise = torch.normal(mean=mean0, std=std).to(device)
                out_decoding = model.AE(batch_x, batch_l, noise)
                out2 = out_decoding[:, :-1]
                loss1 = criterion(out2.reshape(-1, NFEA), batch_x2.reshape(-1))

                Z = model.Enc(batch_x, batch_l)
                pred = model.BinC(Z)
                pred_p = torch.max(F.softmax(pred, dim=1), 1)[1]
                correct_prediction = np.equal(pred_p.cpu().numpy(), batch_p.cpu().numpy())
                correct += np.sum(correct_prediction)
                loss2 = model.BinC.Loss(batch_p, pred_p)
                loss = loss1 + loss2
            print(correct, len(test_loader) * BATCH_SIZE)
            acc = correct / (len(test_loader) * BATCH_SIZE)
            print(f"Test loss: {loss}, Test acc: {acc}")

        # print(f'epoch {epoch}: vae loss is {loss}')
    print('>> Finished Train')


def test(model, criterion, optimizer, scheduler, dataloader, epochs, device):
    model.eval()
    mean0 = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    for i, data in enumerate(dataloader):
        batch_x, batch_l = data
        batch_x = batch_x.to(device)
        batch_l = batch_l.to(device)

        batch_x2 = batch_x[:, 1:]
        b_size = batch_x.shape[0]
