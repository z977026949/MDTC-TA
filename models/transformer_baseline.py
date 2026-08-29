import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model,
            max_len=200
    ):

        super().__init__()

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(
            0,max_len
        ).unsqueeze(1)

        div=torch.exp(
            torch.arange(0,d_model,2)
            *
            (-math.log(10000)/d_model)
        )

        pe[:,0::2]=torch.sin(position*div)

        pe[:,1::2]=torch.cos(position*div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self,x):

        return x+self.pe[:,:x.size(1)]

class TransformerBaseline(nn.Module):

    def __init__(
            self,
            input_dim=4,
            d_model=32,
            nhead=4,
            layers=2,
            dropout=0.1
    ):

        super().__init__()

        self.embedding=nn.Linear(
            input_dim,
            d_model
        )

        self.pos=PositionalEncoding(
            d_model
        )

        encoder=nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=64,

            dropout=dropout,

            batch_first=True

        )

        self.encoder=nn.TransformerEncoder(
            encoder,
            layers
        )

        self.fc=nn.Sequential(

            nn.Linear(d_model,32),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(32,1)

        )

    def forward(self,x):

        x=self.embedding(x)

        x=self.pos(x)

        x=self.encoder(x)

        x=x.mean(dim=1)

        pred=self.fc(x)

        return pred,None,None