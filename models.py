import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Encoder(nn.Module):

    def __init__(self, para, bias=True):
        super(Encoder, self).__init__()

        self.Nseq = para['Nseq']
        self.Nfea = para['Nfea']

        self.hidden_dim = para['hidden_dim']
        self.NLSTM_layer = para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)
        self.encoder_rnn = nn.LSTM(input_size=self.Nfea, hidden_size=self.hidden_dim,
                                   num_layers=self.NLSTM_layer, bias=True,
                                   batch_first=True, bidirectional=False)

        for param in self.encoder_rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, X0, L0):

        batch_size = X0.shape[0]
        device = X0.device
        enc_h0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)
        enc_c0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)

        X = self.embedd(X0)
        out, (encoder_hn, encoder_cn) = self.encoder_rnn(X, (enc_h0, enc_c0))
        last_step_index_list = (L0 - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        Z = out.gather(1, last_step_index_list).squeeze()
        #        Z=torch.sigmoid(Z)
        Z = F.normalize(Z, p=2, dim=1)

        return Z


class Decoder(nn.Module):

    def __init__(self, para, bias=True):
        super(Decoder, self).__init__()

        self.Nseq = para['Nseq']
        self.Nfea = para['Nfea']

        self.hidden_dim = para['hidden_dim']
        self.NLSTM_layer = para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)

        #        self.decoder_rnn = nn.LSTM(input_size=self.Nfea,
        self.decoder_rnn = nn.LSTM(input_size=self.Nfea + self.hidden_dim,
                                   hidden_size=self.hidden_dim, num_layers=self.NLSTM_layer,
                                   bias=True, batch_first=True, bidirectional=False)

        for param in self.decoder_rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        self.decoder_fc1 = nn.Linear(self.hidden_dim, self.Nfea)
        nn.init.xavier_normal_(self.decoder_fc1.weight.data)
        nn.init.normal_(self.decoder_fc1.bias.data)

    def forward(self, Z, X0, L0):

        batch_size = Z.shape[0]
        device = Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)

        X = self.embedd(X0)
        Zm = Z.view(-1, 1, self.hidden_dim).expand(-1, self.Nseq, self.hidden_dim)
        ZX = torch.cat((Zm, X), 2)

        #        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(X0,(Z.view(1,-1,self.hidden_dim),dec_c0))
        dec_out, (decoder_hn, decoder_cn) = self.decoder_rnn(ZX, (dec_h0, dec_c0))
        dec = self.decoder_fc1(dec_out)
        return dec

    def decoding(self, Z):
        batch_size = Z.shape[0]
        device = Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)

        seq = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
        seq[:, 0] = self.Nfea - 2

        #        Xdata_onehot=torch.zeros([batch_size,1,self.Nfea],dtype=torch.float32).to(device)
        #        Xdata_onehot[:,0,self.Nfea-2]=1
        Y = seq
        Zm = Z.view(-1, 1, self.hidden_dim).expand(-1, 1, self.hidden_dim)

        decoder_hn = dec_h0
        decoder_cn = dec_c0
        #        seq2=Xdata_onehot
        for i in range(self.Nseq):
            dec_h0 = decoder_hn
            dec_c0 = decoder_cn

            X = self.embedd(Y)
            ZX = torch.cat((Zm, X), 2)
            dec_out, (decoder_hn, decoder_cn) = self.decoder_rnn(ZX, (dec_h0, dec_c0))
            dec = self.decoder_fc1(dec_out)
            Y = torch.argmax(dec, dim=2)
            #            Xdata_onehot=torch.zeros([batch_size,self.Nfea],dtype=torch.float32).to(device)
            #            Xdata_onehot=Xdata_onehot.scatter_(1,Y,1).view(-1,1,self.Nfea)
            seq = torch.cat((seq, Y), dim=1)
        #            seq2=torch.cat((seq2,dec),dim=1)

        return seq  # , seq2[:,1:]


class Predictor(nn.Module):
    def __init__(self, para, bias=True):
        super(Predictor, self).__init__()
        self.hidden_dim = para['hidden_dim']
        self.fc1 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)

    def forward(self, Z):  # Z encoder之后的embedding
        y1, y2 = self.fc1(Z), self.fc2(Z)
        return y1, y2

    def Loss(self, y_real, y_pred):
        x = y_real - y_pred
        x = x ** 2
        return torch.sum(x) / y_real.shape[0]


class BinaryClassfier(nn.Module):
    def __init__(self, para, bias=True):
        super(BinaryClassfier, self).__init__()
        self.hidden_dim = para['hidden_dim']
        self.fc1 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)

    def forward(self, Z):
        return self.fc1(Z)

    def Loss(self, y_real, y_pred):
        x = y_real - y_pred
        x = x ** 2
        return torch.sum(x) / y_real.shape[0]


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, para, bias=True):
        super(Critic, self).__init__()

        self.hidden_dim = para['hidden_dim']

        self.critic_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc1.weight.data)
        nn.init.normal_(self.critic_fc1.bias.data)

        self.critic_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc2.weight.data)
        nn.init.normal_(self.critic_fc2.bias.data)

        self.critic_fc3 = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.critic_fc3.weight.data)
        nn.init.normal_(self.critic_fc3.bias.data)

    def forward(self, Z0):
        D1 = self.critic_fc1(Z0)
        D1 = torch.relu(D1)
        D2 = self.critic_fc2(D1)
        D2 = torch.relu(D2)
        Dout = self.critic_fc3(D2)

        return Dout

    def clip(self, epsi=0.01):
        torch.clamp_(self.critic_fc1.weight.data, min=-epsi, max=epsi)
        torch.clamp_(self.critic_fc1.bias.data, min=-epsi, max=epsi)
        torch.clamp_(self.critic_fc2.weight.data, min=-epsi, max=epsi)
        torch.clamp_(self.critic_fc2.bias.data, min=-epsi, max=epsi)
        torch.clamp_(self.critic_fc3.weight.data, min=-epsi, max=epsi)
        torch.clamp_(self.critic_fc3.bias.data, min=-epsi, max=epsi)


class Net(nn.Module):

    def __init__(self, para, bias=True):
        super(Net, self).__init__()

        self.Nseq = para['Nseq']
        self.Nfea = para['Nfea']

        self.hidden_dim = para['hidden_dim']
        self.NLSTM_layer = para['NLSTM_layer']

        self.Enc = Encoder(para)
        self.Dec = Decoder(para)

        self.Cri = Critic(para)
        self.Pred = Predictor(para)
        self.BinC = BinaryClassfier(para)
        # self.Gen = Generator(para, self.Pred)

    def AE(self, X0, L0, noise):
        Z = self.Enc(X0, L0)
        #        print(Z.shape, noise.shape)
        Zn = Z + noise
        decoded = self.Dec(Zn, X0, L0)

        return decoded


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
