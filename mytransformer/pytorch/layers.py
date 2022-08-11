import torch
import torch.nn as nn

from .modules import MultiHeadAttention, FeedForwardNetwork



class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model:int=512,
        n_heads:int=8,
        d_ff:int=2048,
        d_k:int=None,
        d_v:int=None,
        dropout:float=0.1) -> None:

        nn.Module.__init__(self)
        self.multi_head_attention = MultiHeadAttention(d_model,n_heads,d_k,d_v)
        self.feed_forward = FeedForwardNetwork(d_model,d_ff)
        self.dropout_1 = nn.Identity()
        self.dropout_2 = nn.Identity()
        if not dropout is None:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
    

    def forward(self,x:torch.Tensor):
        o1 = self.multi_head_attention(x,x,x)
        o1 = self.dropout_1(o1)
        o1 = self.layer_norm_1(x + o1)
        o2 = self.feed_forward(o1)
        o2 = self.dropout_2(o2)
        return self.layer_norm_2(o1 + o2)



class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model:int=512,
        n_heads:int=8,
        d_ff:int=2048,
        d_k:int=None,
        d_v:int=None,
        dropout:float=0.1) -> None:
        
        nn.Module.__init__(self)
        self.multi_head_attention_1 = MultiHeadAttention(d_model,n_heads,d_k,d_v)
        self.multi_head_attention_2 = MultiHeadAttention(d_model,n_heads,d_k,d_v)
        self.feed_forward = FeedForwardNetwork(d_model,d_ff)
        self.dropout_1 = nn.Identity()
        self.dropout_2 = nn.Identity()
        self.dropout_3 = nn.Identity()
        if not dropout is None:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
            self.dropout_3 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)


    def forward(self,x:torch.Tensor,Q_encoder:torch.Tensor,K_encoder:torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones((x.shape[1],x.shape[1])))
        o1 = self.multi_head_attention_1(x,x,x,mask)
        o1 = self.dropout_1(o1)
        o1 = self.layer_norm_1(x + o1)
        o2 = self.multi_head_attention_2(Q_encoder,K_encoder,o1)
        o2 = self.dropout_2(o2)
        o2 = self.layer_norm_2(o1 + o2)
        o3 = self.feed_forward(o2)
        o3 = self.dropout_3(o3)
        return self.layer_norm_3(o2 + o3)
