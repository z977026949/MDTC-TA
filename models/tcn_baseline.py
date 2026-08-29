import torch
import torch.nn as nn


# =====================================================
# Causal Chomp
# =====================================================

class Chomp1d(nn.Module):
    """
    Remove future padding
    Keep causal property
    """

    def __init__(self, chomp_size):
        super().__init__()

        self.chomp_size = chomp_size

    def forward(self, x):

        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size]

# =====================================================
# Temporal Block
# =====================================================

class TemporalBlock(nn.Module):

    """
    Standard TCN residual block

    Conv1D
    +
    Causal padding
    +
    Dilated convolution
    +
    Residual connection
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=1,
            dropout=0.1
    ):

        super().__init__()

        padding = (kernel_size-1)*dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp1 = Chomp1d(
            padding
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp2 = Chomp1d(
            padding
        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.net = nn.Sequential(

            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,

            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2

        )

        if in_channels != out_channels:

            self.downsample = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.downsample = None

        self.final_relu = nn.ReLU()

    def forward(self,x):

        out = self.net(x)

        residual = x

        if self.downsample is not None:

            residual = self.downsample(x)

        return self.final_relu(
            out + residual
        )

# =====================================================
# Pure TCN Baseline
# =====================================================

class TCNBaseline(nn.Module):

    """
    Pure TCN baseline

    Input:

        B,L,F

    Example:

        B,20,4

    Output:

        B,1

    No:

        Gate

        Attention

        Multi-scale

    """

    def __init__(
            self,
            input_dim=4,
            hidden=64,
            dropout=0.1
    ):

        super().__init__()

        self.tcn = nn.Sequential(

            TemporalBlock(
                input_dim,
                hidden,
                kernel_size=3,
                dilation=1,
                dropout=dropout
            ),

            TemporalBlock(
                hidden,
                hidden,
                kernel_size=3,
                dilation=2,
                dropout=dropout
            ),

            TemporalBlock(
                hidden,
                hidden,
                kernel_size=3,
                dilation=4,
                dropout=dropout
            )

        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                hidden,
                32
            ),

            nn.ReLU(),

            nn.Dropout(dropout),


            nn.Linear(
                32,
                1
            )

        )

    def forward(self,x):

        # B,L,F
        # ->
        # B,F,L

        x = x.transpose(1,2)

        x = self.tcn(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        pred = self.fc(x)

        return pred,None,None