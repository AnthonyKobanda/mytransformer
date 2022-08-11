from turtle import forward
import torch
import torch.nn as nn



class ScaleDotProduct(nn.Module):

    def __init__(
        self) -> None:
        
        nn.Module.__init__(self)
        self.softmax = nn.Softmax()


    def forward(
        self,
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor=None) -> torch.Tensor:
        """
        `Q.shape` = [`batch_size`,`n_heads`,`q_len`,`d_k`]\n
        `K.shape` = [`batch_size`,`n_heads`,`k_len`,`d_k`]\n
        `V.shape` = [`batch_size`,`n_heads`,`k_len`,`d_v`]\n
        `mask.shape` = [`q_len`,`v_len`]\n
        """
        attention:torch.Tensor = torch.matmul(Q,K.transpose(dim0=2,dim1=3)) / (K.shape[-1] ** 0.5)
        if not mask is None:
            mask = mask.repeat((Q.shape[0],Q.shape[1],1,1)).to(attention.device)
            attention.masked_fill_(mask==0,float("-inf"))
        # output shape is [batch_size, n_heads, q_len, d_v]
        return torch.matmul(self.softmax(attention), V)



class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model:int=512,
        n_heads:int=8,
        d_k:int=None,
        d_v:int=None) -> None:

        nn.Module.__init__(self)
        if d_k is None: d_k = d_model // n_heads
        if d_v is None: d_v = d_model // n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.Q_projections = nn.ModuleList([nn.Linear(d_model,d_k) for _ in range(n_heads)])
        self.K_projections = nn.ModuleList([nn.Linear(d_model,d_k) for _ in range(n_heads)])
        self.V_projections = nn.ModuleList([nn.Linear(d_model,d_v) for _ in range(n_heads)])
        self.scale_dot_product = ScaleDotProduct()
        self.linear_layer = nn.Linear(n_heads*d_v, d_model)


    def forward(
        self,
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor=None) -> torch.Tensor:
        """
        `Q.shape` = [`batch_size`,`q_len`,`d_model`]\n
        `K.shape` = [`batch_size`,`k_len`,`d_model`]\n
        `V.shape` = [`batch_size`,`k_len`,`d_model`]\n
        `mask.shape` = [`q_len`,`v_len`]\n
        """
        Q_heads = torch.stack([Q_projection(Q) for Q_projection in self.Q_projections],dim=1)
        K_heads = torch.stack([K_projection(K) for K_projection in self.K_projections],dim=1)
        V_heads = torch.stack([V_projection(V) for V_projection in self.V_projections],dim=1)
        o = self.scale_dot_product(Q_heads,K_heads,V_heads,mask).transpose(dim0=1,dim1=2)
        o = o.reshape(Q.shape[0],Q.shape[1],self.n_heads*self.d_v)
        # output shape is [batch_size, n_heads, q_len, d_model]
        return self.linear_layer(o)



class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        d_model:int=512,
        d_ff:int=2048) -> None:
        
        nn.Module.__init__(self)
        self.layer_1 = nn.Linear(d_model,d_ff)
        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(d_ff,d_model)
    

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """
        `x.shape` = [`batch_size`,`q_len`,`d_model`]
        """
        o = self.layer_1(x)
        o = self.activation(o)
        # output shape is [batch_size, n_heads, q_len, d_model]
        return self.layer_2(o)
