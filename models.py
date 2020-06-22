import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    
     ###dropout temporarily taken off 

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]*0.1
        return self.dropout(x)
    
    
class TransformerEncoderV1(nn.Module):

    def __init__(self, inputdim, device, dmodel=256, layers=4, drop=0.0):
        super(TransformerEncoderV1, self).__init__()
        self.device=device
        self.pos_embed_static = PositionalEncoding(d_model=dmodel)
        self.input_embed = nn.Linear(inputdim, dmodel)
        encode_layer = nn.TransformerEncoderLayer(
            d_model=dmodel, nhead=4, dim_feedforward=512, activation='gelu', dropout=drop)
        self.nlayers = nn.TransformerEncoder(encode_layer, num_layers=layers)

    def forward(self, src):
        src = self.input_embed(src)
        src = self.pos_embed_static(src)
        src = self.nlayers(src)
        return src
    
    
    
class TransformerDecoderV1(nn.Module):

    def __init__(self, inputdim, device, dmodel=256,layers=4, drop=0.0):
        super(TransformerDecoderV1, self).__init__()
        self.device=device
        self.pos_embed = PositionalEncoding(d_model=dmodel)
        self.input_embed = nn.Linear(inputdim, dmodel)
        decode_layer = nn.TransformerDecoderLayer(
            d_model=dmodel, nhead=4, dim_feedforward=512, activation='gelu', dropout=drop)
        self.nlayers = nn.TransformerDecoder(decode_layer, num_layers=layers)

    def forward(self, target, memory):
        len_sz = target.shape[0]

        mask = (torch.triu(torch.ones(len_sz, len_sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))

        target = self.input_embed(target)
        target = self.pos_embed(target)
        target = self.nlayers(tgt=target, memory=memory,
                              tgt_mask=mask.to(self.device))

        return target
    
    
class TransformerV1(nn.Module):

    def __init__(self, Encoder, Decoder , outputdim ,dmodel=256):
        super(TransformerV1, self).__init__()
        self.encode = Encoder
        self.decode = Decoder
        self.FC = nn.Sequential(nn.Linear(dmodel, outputdim,bias=True),nn.Sigmoid())

    def forward(self, src, target):
        enc_memory = self.encode(src)
        decoder_result = self.decode(target, enc_memory)
        decoder_result = self.FC(decoder_result)
        return decoder_result


