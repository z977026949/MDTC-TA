import torch
import torch.nn as nn


# =====================================
# Temporal Attention
# =====================================

class TemporalAttention(nn.Module):

    """
    Convolution based Temporal Attention

    Input:
        B,C,T

    Output:
        B,C,T

    Generate temporal weights:
        B,1,T
    """

    def __init__(self, channels):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Conv1d(
                channels,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()

        )


    def forward(self,x):

        weight=self.attn(x)

        return x * weight



# =====================================
# TCN Block
# =====================================

class TemporalBlock(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            dropout
    ):

        super().__init__()


        padding=dilation


        self.conv=nn.Conv1d(

            in_channels,

            out_channels,

            kernel_size=3,

            padding=padding,

            dilation=dilation

        )


        self.relu=nn.ReLU()


        self.dropout=nn.Dropout(
            dropout
        )


        if in_channels!=out_channels:

            self.downsample=nn.Conv1d(

                in_channels,

                out_channels,

                kernel_size=1

            )

        else:

            self.downsample=nn.Identity()



    def forward(self,x):


        residual=self.downsample(x)


        out=self.conv(x)


        # 保持长度一致

        if out.size(-1)!=x.size(-1):

            out=out[:,:,:x.size(-1)]


        out=self.relu(out)

        out=self.dropout(out)


        return self.relu(
            out+residual
        )



# =====================================
# MDTC+ Shared Temporal Attention
# =====================================


class MDTCTA(nn.Module):


    def __init__(

            self,

            input_dim=4,

            hidden=64,

            dropout=0.1

    ):

        super().__init__()



        self.hidden=hidden



        # -----------------------------
        # Multi-scale TCN
        # -----------------------------


        self.tcn1=TemporalBlock(

            input_dim,

            hidden,

            dilation=1,

            dropout=dropout

        )


        self.tcn2=TemporalBlock(

            input_dim,

            hidden,

            dilation=2,

            dropout=dropout

        )


        self.tcn3=TemporalBlock(

            input_dim,

            hidden,

            dilation=4,

            dropout=dropout

        )



        # -----------------------------
        # Shared Temporal Attention
        # -----------------------------

        self.attention=TemporalAttention(

            hidden*3

        )



        # -----------------------------
        #  Feature Transformation
        # -----------------------------


        self.feature_transformation=nn.Sequential(

            nn.Linear(

                hidden*3,

                hidden

            ),

            nn.ReLU(),

            nn.Dropout(dropout)

        )



        # -----------------------------
        # Residual Prediction Head
        # -----------------------------


        self.fc=nn.Linear(

            hidden,

            1

        )



    def forward(self,x):


        # x:
        # B,T,F


        # last Close(t)

        last_close=x[:,-1,3].unsqueeze(1)



        # B,F,T

        x=x.transpose(

            1,

            2

        )



        # -----------------------------
        # Multi-scale features
        # -----------------------------


        f1=self.tcn1(x)

        f2=self.tcn2(x)

        f3=self.tcn3(x)



        # concat channel dimension

        feature=torch.cat(

            [

                f1,

                f2,

                f3

            ],

            dim=1

        )


        # -----------------------------
        # Temporal Attention
        # -----------------------------

        feature=self.attention(

            feature

        )



        # take last time step

        feature=feature[:,:,-1]



        # -----------------------------
        # Transformation
        # -----------------------------


        feature=self.feature_transformation(

            feature

        )



        # predict delta

        delta=self.fc(

            feature

        )



        # residual

        pred=last_close+delta



        return pred,delta,feature