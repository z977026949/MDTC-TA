import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):

    """
    Pure LSTM baseline

    Input:
        B,L,F

    Output:
        B,1
    """

    def __init__(
            self,
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )


        self.fc = nn.Sequential(

            nn.Linear(hidden_dim,16),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(16,1)

        )



    def forward(self,x):

        out,_=self.lstm(x)


        feature=out[:,-1,:]


        pred=self.fc(feature)


        return pred,None,None