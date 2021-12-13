# coding=latin-1
import util201217 as ut
import re, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, FlaubertTokenizer, FlaubertModel
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))


def convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device='cuda:0', floatype=torch.float32):
    nconv = len(conv_channels)
    nfilt = len(filter_sizes)
    return  nn.ModuleList(
            nn.ModuleList(
            nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
            for ifilt in range(nfilt))
            for iconv in range(nconv)).to(device=device, dtype=floatype)


def convs1d_block(embs_lookup, convs, nconv, nfilt, pool_filtersizes=(2, 2, 2, 2), pool_stridesizes=(1, 1, 1, 1)):
    ilastconv = nconv -1
    conveds = [[F.relu(convs[iconv][ifilt](embs_lookup))
                for ifilt in range(nfilt)]
                for iconv in range(nconv)] # [batsize, nr_filters, sent len - filter_sizes[n] + 1]
    pool_filters = [[pool_filtersizes[iconv] if iconv != ilastconv else
                    conveds[iconv][ifilter].shape[2] # poiché voglio che esca un vettore, assegno all'ultimo filtro la stessa dim della colonna in input,
                    for ifilter in range(nfilt)] # così il nr delle colonne in uscita all'ultima conv sarà 1, ed eliminerò la dimensione con squueze.
                    for iconv in range(nconv)]
    pooleds = [[F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=pool_stridesizes[iconv]) if iconv != ilastconv else
                F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=pool_stridesizes[iconv]).squeeze(2)
                for ifilt in range(nfilt)]
                for iconv in range(nconv)] # [batsize, nr_filters]
    concat = torch.cat([pooled for pooled in pooleds[ilastconv]], dim=1) # [batsize, nr_filters * len(filter_sizes)]
    # if torch.isnan(concat).sum() > 0:
    #     print(f"{ut.bcolors.red}Le convuluzioni producono NaN, probabilmente la lunghezzadei filtri eccede quella del testo{ut.bcolors.reset}")
    # for iconv in range(nconv):
    #     for ifilter in range(nfilt):
    #         print("$$$ conveds", iconv, ifilter, conveds[iconv][ifilter], conveds[iconv][ifilter].shape)
    # for iconv in range(nconv):
    #     for ifilter in range(nfilt):
    #         print("$$$ pooleds", iconv, ifilter, pooleds[iconv][ifilter], pooleds[iconv][ifilter].shape)
    # print("$$$ concat", concat.shape)
    return concat


def set_layersizes(insize, outsize, layers):
    return [(insize if i == 1 else int((insize + outsize) * ((layers - i + 1)/layers)), int((insize + outsize) * ((layers - i)/layers)) if i != layers else outsize) for i in range(1, layers+1)]


def load_bert(lang, device):
    config = BertConfig.from_pretrained("bert-base-multilingual-cased",     output_hidden_states=True) if lang == 'ml' else \
             BertConfig.from_pretrained("bert-base-cased",                  output_hidden_states=True) if lang == 'en' else \
             BertConfig.from_pretrained("bert-large-cased",                 output_hidden_states=True) if lang == 'en_large' else \
             BertConfig.from_pretrained("dbmdz/bert-base-italian-cased",    output_hidden_states=True) if lang == 'it' else \
             BertConfig.from_pretrained("dbmdz/bert-base-german-cased",     output_hidden_states=True) if lang == 'de' else \
             BertConfig.from_pretrained("camembert-base",                   output_hidden_states=True) if lang == 'fr' else \
             BertConfig.from_pretrained("pdelobelle/robbert-v2-dutch-base", output_hidden_states=True) if lang == 'nl' else \
             BertConfig.from_pretrained("KB/bert-base-swedish-cased",       output_hidden_states=True) if lang == 'sw' else None
    # config.to_json_file("bert_config.json")
    bert = AutoModel.from_pretrained("bert-base-multilingual-cased",     config=config).to(device=device, dtype=torch.float32) if lang == 'ml' else \
           AutoModel.from_pretrained('bert-base-cased',                  config=config).to(device=device, dtype=torch.float32) if lang == 'en' else \
           AutoModel.from_pretrained("bert-large-cased",                 config=config).to(device=device, dtype=torch.float32) if lang == 'en_large' else \
           AutoModel.from_pretrained('dbmdz/bert-base-italian-cased',    config=config).to(device=device, dtype=torch.float32) if lang == 'it' else \
           AutoModel.from_pretrained('dbmdz/bert-base-german-cased',     config=config).to(device=device, dtype=torch.float32) if lang == 'de' else \
           AutoModel.from_pretrained('camembert-base',                   config=config).to(device=device, dtype=torch.float32) if lang == 'fr' else \
           AutoModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", config=config).to(device=device, dtype=torch.float32) if lang == 'nl' else \
           AutoModel.from_pretrained("KB/bert-base-swedish-cased",       config=config).to(device=device, dtype=torch.float32) if lang == 'sw' else None
    emb_size = bert.config.to_dict()['hidden_size']
    return bert, emb_size, config


class KLregular(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        # Q_pred = F.softmax(Q_pred, dim=1) # pernicioso...!
        Q_pred = F.sigmoid(Q_pred)
        return torch.mean(torch.sum(P_targ * torch.log2(P_targ/Q_pred), dim=1))
        # return torch.sum(P_targ * torch.log2(P_targ / Q_pred))


class KLinverse(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        # Q_pred = F.softmax(Q_pred, dim=1) # pernicioso...!
        Q_pred = F.sigmoid(Q_pred)
        return torch.mean(torch.sum(Q_pred * torch.log2(Q_pred/P_targ), dim=1))
        # return torch.sum(Q_pred * torch.log2(Q_pred / P_targ))


class CrossEntropySoft(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        Q_pred = F.softmax(Q_pred, dim=1) # per allinearmi a nn.CrossEntropyLoss, che applica softmax a valori qualsiasi
        return torch.mean(-torch.sum(P_targ * torch.log(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


# Samuel Lynn-Evans
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512, device='cuda:0'):
        self.device = device
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        # cioè sequence_length (position) ed emb_size
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
# If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer,
# you should register them as buffers. Buffers won?t be returned in model.parameters(), so that the optimizer won?t have a chance to update them.

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # make embeddings relatively larger
        seq_len = x.size(1) # add constant to embedding
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(device=self.device) # Variable ammette la back propagation
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device='cuda:0', dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model).to(device=device)
        self.v_linear = nn.Linear(d_model, d_model).to(device=device)
        self.k_linear = nn.Linear(d_model, d_model).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model).to(device=device)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, device='cuda:0', d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device=device)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, device='cuda:0', eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size).to(device=device))
        self.bias = nn.Parameter(torch.zeros(self.size).to(device=device))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model, device)
        self.norm_2 = Norm(d_model, device)
        self.attn = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn.forward(x2, x2, x2, mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.ff.forward(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, rep_size, n_heads, n_layers, max_seq_len=512, device='cuda:0'):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoder(rep_size, max_seq_len=max_seq_len, device=device)
        self.layers = get_clones(EncoderLayer(rep_size, n_heads, device), n_layers)
        self.norm = Norm(rep_size, device)

    def forward(self, src, mask=None):
        x = self.pe.forward(src)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm.forward(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn_1.forward(x2, x2, x2, trg_mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.attn_2.forward(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3.forward(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, device='cuda:0'):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, device)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed.forward(trg)
        x = self.pe.forward(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm.forward(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder.forward(src, src_mask)
        d_output = self.decoder.forward(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output




class StlVecFc(nn.Module):
    def __init__(self, emb_size, y0_size, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        for layer in self.fc_layers:
            embs = layer(embs)
            # print(embs.shape, "$")
        out = embs.squeeze(1)
        out = self.dropout(out)
        out = F.sigmoid(out)
        return out


class Mtl1VecFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2VecFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)
        
        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        return y0_out, y1_out, y2_out


class Mtl3VecFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)

        y3_layer = embs
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)

        return y0_out, y1_out, y2_out, y3_out


class Mtl6VecFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(emb_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(emb_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(emb_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)

        y3_layer = embs
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)

        y4_layer = embs
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)

        y5_layer = embs
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)

        y6_layer = embs
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)

        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out




class StlVecTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        embs = self.encoder.forward(embs.unsqueeze(1)).squeeze(1)

        for layer in self.fc_layers:
            embs = layer(embs)

        out = embs.squeeze(1)
        out = self.dropout(out)
        out = F.sigmoid(out)
        return out


class Mtl1VecTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        embs = self.encoder.forward(embs.unsqueeze(1)).squeeze(1)

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        return y0_out, y1_out


class Mtl2VecTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        embs = self.encoder.forward(embs.unsqueeze(1)).squeeze(1)

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)

        return y0_out, y1_out, y2_out


class Mtl3VecTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        embs = self.encoder.forward(embs.unsqueeze(1)).squeeze(1)

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)

        y3_layer = embs
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)

        return y0_out, y1_out, y2_out, y3_out


class Mtl6VecTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(emb_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(emb_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(emb_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, embs):
        embs = self.encoder.forward(embs.unsqueeze(1)).squeeze(1)

        y0_layer = embs
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = embs
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = self.dropout(y1_layer)

        y2_layer = embs
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)

        y3_layer = embs
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)

        y4_layer = embs
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)

        y5_layer = embs
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)

        y6_layer = embs
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)

        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out




class StlIds4lookupConvFc(nn.Module):
    def __init__(self, emb, y0_size, trainable=False, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        return y0_out


class Mtl1Ids4lookupConvFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, trainable=False, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2Ids4lookupConvFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, trainable=False, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        return y0_out, y1_out, y2_out


class Mtl3Ids4lookupConvFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, y3_size, trainable=False, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(convout_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = out_conv
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        return y0_out, y1_out, y2_out, y3_out


class Mtl6Ids4lookupConvFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, trainable=False, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(convout_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(convout_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(convout_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(convout_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = out_conv
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        y4_layer = out_conv
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)
        
        y5_layer = out_conv
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)
        
        y6_layer = out_conv
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)
        
        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out




class StlLookupConvFc(nn.Module):
    def __init__(self, emb_size, y0_size, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        return y0_out


class Mtl1LookupConvFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)
        return y0_out, y1_out
   
        
class Mtl2LookupConvFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        return y0_out, y1_out, y2_out
   
        
class Mtl3LookupConvFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(convout_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = out_conv
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        return y0_out, y1_out, y2_out, y3_out
   
        
class Mtl6LookupConvFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, conv_channels=(16, 32), filter_sizes=(3, 4, 5), conv_stridesizes=(1, 1), pool_filtersizes=(2, 2), pool_stridesizes=(1, 1), fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.convs = convs1d(emb_size, conv_channels, filter_sizes, conv_stridesizes, device, floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        y0_layersizes = set_layersizes(convout_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(convout_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(convout_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(convout_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(convout_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(convout_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(convout_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = lookup.permute(0, 2, 1)
        out_conv = convs1d_block(lookup, self.convs, self.nconv, self.nfilt, self.pool_filtersizes, self.pool_stridesizes)
        out_conv = self.dropout(out_conv)

        y0_layer = out_conv
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out_conv
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = out_conv
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = out_conv
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        y4_layer = out_conv
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)
        
        y5_layer = out_conv
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)
        
        y6_layer = out_conv
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)
        
        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out
   
        


class StlIds4lookupTransFc(nn.Module):
    def __init__(self, emb, y0_size, trainable=False, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup  = self.embs(text)

        lookup = self.encoder.forward(lookup, mask.unsqueeze(1)).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        return y0_out


class Mtl1Ids4lookupTransFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, trainable=False, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup  = self.embs(text)
        
        lookup = self.encoder.forward(lookup, mask.unsqueeze(1)).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2Ids4lookupTransFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, trainable=False, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup  = self.embs(text)
        
        lookup = self.encoder.forward(lookup, mask.unsqueeze(1)).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        return y0_out, y1_out, y2_out


class Mtl3Ids4lookupTransFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, y3_size, trainable=False, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup  = self.embs(text)
        
        lookup = self.encoder.forward(lookup, mask.unsqueeze(1)).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = lookup
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        return y0_out, y1_out, y2_out, y3_out


class Mtl6Ids4lookupTransFc(nn.Module):
    def __init__(self, emb, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, trainable=False, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        if type(emb) == np.ndarray:
            emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(emb_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(emb_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(emb_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup  = self.embs(text)
        
        lookup = self.encoder.forward(lookup, mask.unsqueeze(1)).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = lookup
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        y4_layer = lookup
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)
        
        y5_layer = lookup
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)
        
        y6_layer = lookup
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)
        
        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out




class StlLookupTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = self.encoder.forward(lookup).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        return y0_out


class Mtl1LookupTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = self.encoder.forward(lookup).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2LookupTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = self.encoder.forward(lookup).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        return y0_out, y1_out, y2_out


class Mtl3LookupTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = self.encoder.forward(lookup).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = lookup
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        return y0_out, y1_out, y2_out, y3_out


class Mtl6LookupTransFc(nn.Module):
    def __init__(self, emb_size, y0_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        
        self.encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(emb_size, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(emb_size, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(emb_size, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y3_layersizes = set_layersizes(emb_size, y3_size, fc_layers)
        self.y3_fc_layers = nn.ModuleList(nn.Linear(y3_layersizes[il][0], y3_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y4_layersizes = set_layersizes(emb_size, y4_size, fc_layers)
        self.y4_fc_layers = nn.ModuleList(nn.Linear(y4_layersizes[il][0], y4_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y5_layersizes = set_layersizes(emb_size, y5_size, fc_layers)
        self.y5_fc_layers = nn.ModuleList(nn.Linear(y5_layersizes[il][0], y5_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y6_layersizes = set_layersizes(emb_size, y6_size, fc_layers)
        self.y6_fc_layers = nn.ModuleList(nn.Linear(y6_layersizes[il][0], y6_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)

        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, lookup):
        lookup = self.encoder.forward(lookup).mean(1)

        y0_layer = lookup
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer
        y0_out = y0_out.squeeze(1)
        y0_out = F.sigmoid(y0_out)

        y1_layer = lookup
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer
        y1_out = self.dropout(y1_out)

        y2_layer = lookup
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = self.dropout(y2_layer)
        
        y3_layer = lookup
        for layer in self.y3_fc_layers:
            y3_layer = layer(y3_layer)
        y3_out = self.dropout(y3_layer)
        
        y4_layer = lookup
        for layer in self.y4_fc_layers:
            y4_layer = layer(y4_layer)
        y4_out = self.dropout(y4_layer)
        
        y5_layer = lookup
        for layer in self.y5_fc_layers:
            y5_layer = layer(y5_layer)
        y5_out = self.dropout(y5_layer)
        
        y6_layer = lookup
        for layer in self.y6_fc_layers:
            y6_layer = layer(y6_layer)
        y6_out = self.dropout(y6_layer)
        
        return y0_out, y1_out, y2_out, y3_out, y4_out, y5_out, y6_out




class StlTrainBertMatHcatHierTransFc(nn.Module):
    def __init__(self, lang,  y_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_txt_fc = nr_wrd_in_txt * txt_fc_outsize

        self.doc_encoder = Encoder(reshaped_txt_fc, att_heads, att_layers, device=device)

        doc_layersizes = set_layersizes(reshaped_txt_fc, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_doc_fc = nr_txt_in_doc * doc_fc_outsize

        out_layersizes = set_layersizes(reshaped_doc_fc, y_size, fc_layers)
        self.out_fc_layers = nn.ModuleList(nn.Linear(out_layersizes[il][0], out_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
            # txt = self.bert(input_ids=text.view(text.shape[0], -1), attention_mask=mask.view(mask.shape[0], -1))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)

        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], txt.shape[1], -1)
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)
        for layer in self.out_fc_layers:
            out = layer(out)

        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out


class Mtl1TrainBertMatHcatHierTransFc(nn.Module):
    def __init__(self, lang,  y0_size, y1_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_txt_fc = nr_wrd_in_txt * txt_fc_outsize

        self.doc_encoder = Encoder(reshaped_txt_fc, att_heads, att_layers, device=device)

        doc_layersizes = set_layersizes(reshaped_txt_fc, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_doc_fc = nr_txt_in_doc * doc_fc_outsize

        y0_layersizes = set_layersizes(reshaped_doc_fc, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(reshaped_doc_fc, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
            # txt = self.bert(input_ids=text.view(text.shape[0], -1), attention_mask=mask.view(mask.shape[0], -1))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)

        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], txt.shape[1], -1)
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)

        y0_layer = out
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2TrainBertMatHcatHierTransFc(nn.Module):
    def __init__(self, lang,  y0_size, y1_size, y2_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_txt_fc = nr_wrd_in_txt * txt_fc_outsize

        self.doc_encoder = Encoder(reshaped_txt_fc, att_heads, att_layers, device=device)

        doc_layersizes = set_layersizes(reshaped_txt_fc, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_doc_fc = nr_txt_in_doc * doc_fc_outsize

        y0_layersizes = set_layersizes(reshaped_doc_fc, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(reshaped_doc_fc, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(reshaped_doc_fc, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
            # txt = self.bert(input_ids=text.view(text.shape[0], -1), attention_mask=mask.view(mask.shape[0], -1))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)

        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], txt.shape[1], -1)
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)

        y0_layer = out
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)

        y2_layer = out
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = y2_layer.squeeze(1)
        y2_out = self.dropout(y2_out)

        return y0_out, y1_out, y2_out




class StlTrainBertMatVcatHierTransFc(nn.Module):
    def __init__(self, lang,  y_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.doc_encoder = Encoder(txt_fc_outsize, att_heads, att_layers, max_seq_len=nr_wrd_in_txt * nr_txt_in_doc, device=device)

        doc_layersizes = set_layersizes(txt_fc_outsize, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        out_layersizes = set_layersizes(doc_fc_outsize * nr_wrd_in_txt * nr_txt_in_doc, y_size, fc_layers)
        self.out_fc_layers = nn.ModuleList(nn.Linear(out_layersizes[il][0], out_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], -1, txt.shape[3])
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)
        for layer in self.out_fc_layers:
            out = layer(out)

        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out


class Mtl1TrainBertMatVcatHierTransFc(nn.Module):
    def __init__(self, lang,  y0_size, y1_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.doc_encoder = Encoder(txt_fc_outsize, att_heads, att_layers, max_seq_len=nr_wrd_in_txt * nr_txt_in_doc, device=device)

        doc_layersizes = set_layersizes(txt_fc_outsize, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_doc_fc = doc_fc_outsize * nr_wrd_in_txt * nr_txt_in_doc

        y0_layersizes = set_layersizes(reshaped_doc_fc, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(reshaped_doc_fc, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], -1, txt.shape[3])
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)

        y0_layer = out
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)

        return y0_out, y1_out


class Mtl2TrainBertMatVcatHierTransFc(nn.Module):
    def __init__(self, lang,  y0_size, y1_size, y2_size, nr_wrd_in_txt, nr_txt_in_doc, trainable=False, att_heads=1, att_layers=1, fc_layers=1, txt_fc_outsize=10, doc_fc_outsize=10, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable
        self.bert, embsize, _ = load_bert(lang, device)

        txt_layersizes = set_layersizes(embsize, txt_fc_outsize, fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.doc_encoder = Encoder(txt_fc_outsize, att_heads, att_layers, max_seq_len=nr_wrd_in_txt * nr_txt_in_doc, device=device)

        doc_layersizes = set_layersizes(txt_fc_outsize, doc_fc_outsize, fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(fc_layers)).to(device=device)
        reshaped_doc_fc = doc_fc_outsize * nr_wrd_in_txt * nr_txt_in_doc

        y0_layersizes = set_layersizes(reshaped_doc_fc, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(reshaped_doc_fc, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(reshaped_doc_fc, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            txt = self.bert(input_ids=text.view(-1, text.shape[2]), attention_mask=mask.view(-1, mask.shape[2]))[0].view(text.shape[0], text.shape[1], text.shape[2], -1)
        for layer in self.txt_fc_layers:
            txt = layer(txt)

        doc = txt.view(txt.shape[0], -1, txt.shape[3])
        doc = self.doc_encoder.forward(doc)

        for layer in self.doc_fc_layers:
            doc = layer(doc)

        out = doc.view(txt.shape[0], -1)

        y0_layer = out
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = out
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)

        y2_layer = out
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = y2_layer.squeeze(1)
        y2_out = self.dropout(y2_out)

        return y0_out, y1_out, y2_out




class StlVecHierTransMean(nn.Module): # la media perde troppa info
    def __init__(self, emb_size, y0_size, att_heads=1, att_layers=1, out_fc_layers=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()

        self.doc_encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        out_layersizes = set_layersizes(emb_size, y0_size, out_fc_layers)
        self.out_fc_layers = nn.ModuleList(nn.Linear(out_layersizes[il][0], out_layersizes[il][1]) for il in range(out_fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text):
        doc = self.doc_encoder.forward(text).mean(1) # [batsize, embsize]

        for layer in self.out_fc_layers:
            doc = layer(doc)
            # print(out.shape, "$")

        doc = self.dropout(doc)
        doc = F.sigmoid(doc).squeeze(1)
        return doc


class StlVecHierTransCat(nn.Module): # la media perde troppa info
    def __init__(self, cont_size, emb_size, y0_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.doc_encoder = Encoder(emb_size, att_heads, att_layers, device=device)

        out_layersizes = set_layersizes(emb_size * cont_size, y0_size, fc_layers)
        self.fc_layers = nn.ModuleList(nn.Linear(out_layersizes[il][0], out_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text):
        # print()
        # print(text.shape, '$')
        doc = self.doc_encoder.forward(text)
        # print(doc.shape, '$$')
        doc = doc.view(doc.shape[0], -1)
        # print(doc.shape, '$$$')

        for layer in self.fc_layers:
            doc = layer(doc)
            # print(doc.shape, "$")

        doc = self.dropout(doc)
        doc = F.sigmoid(doc).squeeze(1)
        return doc


class Mtl1VecHierTransCat(nn.Module): # la media perde troppa info
    def __init__(self, nr_docs, embsize, y0_size, y1_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.doc_encoder = Encoder(embsize, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(embsize * nr_docs, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(embsize * nr_docs, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text):
        doc = self.doc_encoder.forward(text) # [batsize, embsize]
        doc = doc.view(doc.shape[0], -1)

        y0_layer = doc
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = doc
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)
        return y0_out, y1_out


class Mtl2VecHierTransCat(nn.Module): # la media perde troppa info
    def __init__(self, nr_docs, embsize, y0_size, y1_size, y2_size, att_heads=1, att_layers=1, fc_layers=1, droprob=.1, device='cuda:0'):
        super().__init__()

        self.doc_encoder = Encoder(embsize, att_heads, att_layers, device=device)

        y0_layersizes = set_layersizes(embsize * nr_docs, y0_size, fc_layers)
        self.y0_fc_layers = nn.ModuleList(nn.Linear(y0_layersizes[il][0], y0_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y1_layersizes = set_layersizes(embsize * nr_docs, y1_size, fc_layers)
        self.y1_fc_layers = nn.ModuleList(nn.Linear(y1_layersizes[il][0], y1_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        y2_layersizes = set_layersizes(embsize * nr_docs, y2_size, fc_layers)
        self.y2_fc_layers = nn.ModuleList(nn.Linear(y2_layersizes[il][0], y2_layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text):
        doc = self.doc_encoder.forward(text) # [batsize, embsize]
        doc = doc.view(doc.shape[0], -1)

        y0_layer = doc
        for layer in self.y0_fc_layers:
            y0_layer = layer(y0_layer)
        y0_out = y0_layer.squeeze(1)
        y0_out = self.dropout(y0_out)
        y0_out = F.sigmoid(y0_out)

        y1_layer = doc
        for layer in self.y1_fc_layers:
            y1_layer = layer(y1_layer)
        y1_out = y1_layer.squeeze(1)
        y1_out = self.dropout(y1_out)

        y2_layer = doc
        for layer in self.y2_fc_layers:
            y2_layer = layer(y2_layer)
        y2_out = y2_layer.squeeze(1)
        y2_out = self.dropout(y2_out)

        return y0_out, y1_out, y2_out








