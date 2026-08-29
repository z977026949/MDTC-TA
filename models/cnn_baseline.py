import torch
import torch.nn as nn



class CNN1DBaseline(nn.Module):


    def __init__(
            self,
            input_dim=4,
            hidden=64,
            dropout=0.1
    ):

        super().__init__()


        self.conv=nn.Sequential(

            nn.Conv1d(
                input_dim,
                hidden,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Dropout(dropout),


            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)

        )


        self.fc=nn.Sequential(

            nn.Linear(hidden,16),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(16,1)

        )


    def forward(self,x):

        # B,L,F
        x=x.transpose(1,2)

        x=self.conv(x)

        x=x.squeeze(-1)


        pred=self.fc(x)


        return pred,None,None