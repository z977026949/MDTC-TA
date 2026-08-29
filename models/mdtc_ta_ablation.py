import torch
import torch.nn as nn


# =====================================================
# Temporal Attention
# 与 MDTCTA 保持一致
# =====================================================

class TemporalAttention(nn.Module):

    """
    Shared Temporal Attention

    Input:
        B,C,T

    Output:
        B,C,T
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

        weight = self.attn(x)

        return x * weight



# =====================================================
# TCN Block
# 与 MDTCTA 保持一致
# =====================================================

class TemporalBlock(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            dropout
    ):

        super().__init__()


        self.conv = nn.Conv1d(

            in_channels,

            out_channels,

            kernel_size=3,

            padding=dilation,

            dilation=dilation

        )


        self.relu = nn.ReLU()


        self.dropout = nn.Dropout(
            dropout
        )


        if in_channels != out_channels:

            self.downsample = nn.Conv1d(

                in_channels,

                out_channels,

                kernel_size=1

            )

        else:

            self.downsample = nn.Identity()



    def forward(self,x):


        residual = self.downsample(x)


        out = self.conv(x)


        # 保持长度一致

        if out.size(-1) != x.size(-1):

            out = out[:, :, :x.size(-1)]


        out = self.relu(out)

        out = self.dropout(out)


        return self.relu(
            out + residual
        )



# =====================================================
# MDTC-TA Ablation
# =====================================================

class MDTCTAAblation(nn.Module):


    def __init__(

            self,

            mode="full",

            input_dim=4,

            hidden=64,

            dropout=0.1

    ):

        super().__init__()


        self.mode = mode


        self.hidden = hidden



        # =================================================
        # Multi-scale TCN
        # =================================================


        self.tcn1 = TemporalBlock(

            input_dim,

            hidden,

            dilation=1,

            dropout=dropout

        )


        if mode != "single_scale":


            self.tcn2 = TemporalBlock(

                input_dim,

                hidden,

                dilation=2,

                dropout=dropout

            )


            self.tcn3 = TemporalBlock(

                input_dim,

                hidden,

                dilation=4,

                dropout=dropout

            )


            self.multi_scale=True


        else:

            self.multi_scale=False



        # =================================================
        # Shared Temporal Attention
        # =================================================


        if mode != "no_attention":


            channel_dim = (

                hidden*3

                if self.multi_scale

                else hidden

            )


            self.attention = TemporalAttention(

                channel_dim

            )


            self.use_attention=True


        else:

            self.use_attention=False



        # =================================================
        # Transformation
        # 与主模型完全一致
        # =================================================


        if mode != "no_transformation":


            transformation_input_dim = (

                hidden*3

                if self.multi_scale

                else hidden

            )

            self.feature_transformation = nn.Sequential(

                nn.Linear(

                    transformation_input_dim,

                    hidden

                ),

                nn.ReLU(),

                nn.Dropout(dropout)

            )


            output_dim = hidden


        else:


            transformation_input_dim = (

                hidden*3

                if self.multi_scale

                else hidden

            )


            self.feature_transformation = nn.Identity()


            output_dim = transformation_input_dim



        # =================================================
        # Prediction Head
        # =================================================


        self.fc = nn.Linear(

            output_dim,

            1

        )



    def forward(self,x):


        # x:
        # B,T,F


        # 当前close

        last_close = x[:,-1,3].unsqueeze(1)



        # B,F,T

        x = x.transpose(

            1,

            2

        )

        # =================================================
        # Multi-scale TCN
        # =================================================

        f1 = self.tcn1(x)

        features = [f1]

        if self.multi_scale:
            f2 = self.tcn2(x)

            f3 = self.tcn3(x)

            features.append(f2)

            features.append(f3)

        # =================================================
        # Feature Concatenation
        # =================================================

        feature = torch.cat(

            features,

            dim=1

        )


        # =================================================
        # Shared Temporal Attention
        # =================================================


        if self.use_attention:
            feature = self.attention(feature)

        # =================================================
        # Take last time step
        # =================================================

        feature = feature[:, :, -1]

        # =================================================
        # Transformation
        # =================================================

        feature = self.feature_transformation(

            feature

        )



        # =================================================
        # Delta prediction
        # =================================================


        delta = self.fc(

            feature

        )



        # =================================================
        # Residual prediction
        # =================================================


        if self.mode != "no_residual":

            pred = last_close + delta


        else:

            pred = delta



        return pred, delta, feature