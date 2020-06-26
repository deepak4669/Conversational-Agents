import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import copy
import math




device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

class EncoderDecoder(nn.Module):
    
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super().__init__()
        
        self.encoder=encoder
        self.decoder=decoder
        
        self.source_embed=source_embed
        self.target_embed=target_embed
        
        self.generator=generator # Linear + Log_softmax
        
    def forward(self,source,target,source_mask,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)
    
    def encode(self,source,source_mask):
        return self.encoder(self.source_embed(source),source_mask)
    
    def decode(self,memory, source_mask,target,target_mask):
        return self.decoder(self.target_embed(target),memory,source_mask,target_mask)

class Generator(nn.Module):
    
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.projection=nn.Linear(d_model,vocab_size)
        
    def forward(self,decoder_output):
        return F.log_softmax(self.projection(decoder_output),dim=-1)


def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    
    def __init__(self,layer,N):
        super().__init__()
        
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)
    
    def forward(self,x,mask):
        
        for layer in self.layers:
            x=layer(x,mask)
        
        return self.norm(x)



class LayerNorm(nn.Module):
    
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps
        
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2



class SublayerConnection(nn.Module):
    
    def __init__(self,size,dropout):
        super().__init__()
        
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNorm(size)
        
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super().__init__()
        
        self.attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size
        
    def forward(self,x,mask):
        
        x=self.sublayer[0](x,lambda x: self.attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)


class Decoder(nn.Module):
    
    def __init__(self,layer,N):
        super().__init__()
        
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)
    
    def forward(self,x,memory,curr_mask,tgt_mask):
        
        for layer in self.layers:
            x=layer(x,memory,curr_mask,tgt_mask)
            
        return self.norm(x)

class DecoderLayer(nn.Module):
    
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        
        self.sublayer=clones(SublayerConnection(size,dropout),3)
        
    def forward(self,x,memory,src_mask,tgt_mask):
        
        m=memory
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)



def attention(query,key,value,mask=None,dropout=None):
    
    d_k=query.size(-1)

    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
        
    p_attn=F.softmax(scores,dim=-1)
    
    if dropout is not None:
        p_attn=dropout(p_attn)
        
    return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    
    def __init__(self,h,d_model,dropout=0.1):
        super().__init__()
        
        assert d_model%h==0
        
        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,query,key,values,mask=None):
        
        if mask is not None:
            mask=mask.unsqueeze(1)
            
        nbatches=query.size(0)
        
        query,key,values=[l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l, x in zip(self.linears,(query,key,values))]
        
        x,self.attn=attention(query,key,values,mask=mask,dropout=self.dropout)
        
        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Embeddings(nn.Module):
    
    def __init__(self,d_model,vocab):
        super().__init__()
        
        self.embed=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    
    def forward(self,x):
#         print(x.device)
        return self.embed(x)*math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    
    def __init__(self,d_model,dropout,max_len=5000):
        super().__init__()
        
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model,dtype=torch.float)
        position=torch.arange(0.,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0.,d_model,2)*-(math.log(10000.0)/d_model))
        
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)



"""
triu function generates a copy of matrix with elemens below kth diagonal zeroed.
The main diagonal is zeroeth diagonal above is first(k=1) and so on.

Eg:
A=[[1,2,3],[4,5,6],[7,8,9]]
for above matrix:
triu(A,k=1)
will give [[0,2,3],[0,0,6],[0,0,0]]
"""

def subsequent_mask(size):
    attn_shape=(1,size,size)
    mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    
    return torch.from_numpy(mask)==0



def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    
    c=copy.deepcopy
    attn=MultiHeadedAttention(h,d_model)
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    model=EncoderDecoder(Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
                        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
                        nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
                        nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
                        Generator(d_model,tgt_vocab))
    
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model


sample_model=make_model(7688,7688,1,512,2048,8,0.1)
sample_model.to(device)


#Sample Run
source=torch.ones(5,12,dtype=torch.long,device=device)
target=torch.ones(5,12,dtype=torch.long,device=device)
source_mask=torch.ones(5,12,12,dtype=torch.long,device=device)
target_mask=torch.ones(5,12,12,dtype=torch.long,device=device)
out=sample_model(source,target,source_mask,target_mask)
print("-"*80)
print("Output size: "+str(out.shape))
print("-"*80)

