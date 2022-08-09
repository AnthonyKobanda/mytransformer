from turtle import forward
import torch
import torch.nn as nn

from .layers import EncoderLayer, DecoderLayer



class Transformer(nn.Module):

    def __init__(
        self,
        d_output:int,
        n_encoder_layers:int=6,
        n_decoder_layers:int=6,
        d_model:int=512,
        n_heads:int=8,
        d_ff:int=2048,
        d_k:int=None,
        d_v:int=None,
        dropout:float=0.1) -> None:

        nn.Module.__init__(self)
        self.encoders = nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,d_k,d_v,dropout) for _ in range(n_encoder_layers)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,d_k,d_v,dropout) for _ in range(n_decoder_layers)])
        self.linear_layer = nn.Linear(d_model,d_output)

    
    def forward(
        self,
        x:torch.Tensor) -> torch.Tensor:

        o1 = x
        for encoder in self.encoders:
            o1 = encoder(o1)
        o2 = o1
        for decoder in self.decoders:
            o2 = decoder(o1,o1,o2)
        return self.linear_layer(o2)
