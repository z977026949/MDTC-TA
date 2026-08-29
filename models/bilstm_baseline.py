import torch
import torch.nn as nn


class BiLSTMBaseline(nn.Module):

    def __init__(
            self,
            input_dim=4,
            hidden_dim=64,
            num_layers=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.fc = nn.Linear(
            hidden_dim*2,
            1
        )

    def forward(self,x):

        """
        x:
        [batch,window,features]

        """

        out,_=self.lstm(x)


        # 取最后时间步
        out=out[:,-1,:]


        pred=self.fc(out)


        return pred,None,None