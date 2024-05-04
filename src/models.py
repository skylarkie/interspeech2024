import torch
from torch import nn
import torch.nn.functional as F
EPS = torch.finfo(torch.float32).eps

def _padded_cat(x, y, dim=1):
    x_pad = F.pad(x, (0, y.shape[3] - x.shape[3], 
                      0, y.shape[2] - x.shape[2]))
    z = torch.cat((x_pad, y), dim=dim)
    return z

def apply_mask_bound(mask, mask_bound=None):
    # apply mask bounding
    if mask_bound == "sigmoid":
        mask = torch.sigmoid(mask)
    elif mask_bound == "tanh":
        mask = torch.tanh(mask)
    else:
        raise ValueError(f"Mask bound error {mask_bound}")
    return mask

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), 
                 padding=(0, 1), causal=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.PReLU()
        self.causal = causal
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]
        x = self.norm(x)
        x = self.activation(x)
        return x


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1),
                 padding=(0, 0), output_padding=(0, 0), is_last=False, causal=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                       padding, output_padding)
        self.is_last = is_last
        self.causal = causal
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x

class DPConformer(nn.Module):
    """
    Dual-path Conformer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.debug = kwargs.pop("debug", False)
        num_layers = kwargs.pop("num_layers", 4)
        encoder_dim = kwargs.pop("encoder_dim", 64)
        _ = kwargs.pop('bidirectional', False) # remove item from kwargs
        kwargs.update({
            "encoder_dim": encoder_dim,
            "num_attention_heads": kwargs.pop("num_attention_heads", 4),
            "conv_kernel_size": kwargs.pop("conv_kernel_size", 33)
            })
        for _ in range(num_layers):
            self.row_rnn.append(ConformerBlock(**kwargs))
            self.col_rnn.append(ConformerBlock(**kwargs))
            self.row_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply rnn on dim1 first and then dim2
        
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
                    
        return output

from cbam import CBAM
class EnhancingModel(torch.nn.Module):
    # Triple UNet
      def __init__(self, param, in_channels=4, **kwargs):
            super().__init__()
            self.debug = kwargs.get('debug', False)
            self.mask = kwargs.get('mask', True)
            self.homo_mask = kwargs.get('homo_mask', True)
            self.mask_bound = kwargs.get('mask_bound', 'tanh,sigmoid,sigmoid').split(",")
            norm = kwargs.get('norm', 'batch')
            block_att = kwargs.get('block_att', True)
            self.homo_w = kwargs.get('homo_w', 0.5)
            
            param["encoder"][0][0] = in_channels        
            self.encoder = nn.ModuleList([ConvBlock(*item, norm=norm) for item in param["encoder"]])        
      
            param["decoder"][-1][1] = 2
            self.decoder1 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param["decoder"]])        
            param["decoder"][-1][1] = 1
            self.decoder2 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param["decoder"]])
            self.decoder3 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param["decoder"]])        
            self.block_att = nn.ModuleList([
                  CBAM(item[1]) if block_att is True else nn.Identity() for item in param["encoder"]])
            
            rnn_block = kwargs.get('rnn_block', 'DPRNN')
            rnn_kwargs = {
            "encoder_dim": param["encoder"][-1][1],
            "num_layers": kwargs.get('rnn_layers', 4)
            }

            self.rnn_block = DPConformer(**rnn_kwargs)

      def forward(self, vocal, excit, phase):
            e = torch.cat((phase, vocal, excit), dim=1)
            e_list = []
            for i, layer in enumerate(self.encoder):
                  e = layer(e)
                  e = self.block_att[i](e)
                  e_list.append(e)
                  if self.debug:
                    print(f"encoder_{i}: {e.shape}")
            rnn_out = self.rnn_block(e)
            idx = len(e_list)
            d1 = d2 = d3 = rnn_out
            if self.debug:
                  print(f"rnn_out: {rnn_out.shape}")
            for i, (d1_layer, d2_layer, d3_layer) in enumerate(zip(self.decoder1, self.decoder2, self.decoder3)):
                  idx = idx - 1
                  d1 = d1_layer(_padded_cat(d1, e_list[idx]))
                  d2 = d2_layer(_padded_cat(d2, e_list[idx]))
                  d3 = d3_layer(_padded_cat(d3, e_list[idx]))
                  if self.debug:
                    print(f"d1_{i}: {d1.shape}, d2_{i}: {d2.shape}, d3_{i}: {d3.shape}")        
            if self.mask is True:
                  d1 = apply_mask_bound(d1, self.mask_bound[0])
            if self.homo_mask is True:
                  d2 = apply_mask_bound(d2, self.mask_bound[1])
                  d3 = apply_mask_bound(d3, self.mask_bound[2])
            if self.homo_w == 1:
                  d1 = None
            elif self.homo_w == 0:
                  d2 = d3 = None
            est_phase = phase + d1
            est_phase = est_phase / (torch.norm(est_phase, dim=1, keepdim=True) + 1e-5)
            est_vocal = d2 * vocal
            est_excit = d3 * excit
            return est_vocal, est_excit, est_phase

class PhaseDisc(nn.Module):    
    def __init__(self, param, debug=False):
        super(PhaseDisc, self).__init__()
        self.debug = debug
        self.encoder = nn.ModuleList([ConvBlock(*item) for item in param["encoder"]])
        self.lstm_layer = nn.LSTM(*param["lstm"])
        self.decoder = nn.ModuleList([TransConvBlock(*item) for item in param["decoder"]])

    def forward(self, x):
        self.lstm_layer.flatten_parameters()
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
            if self.debug:
                print(f"encoder_{i}: {e.shape}")   

        batch_size, n_channels, n_freq, n_frames = e.shape

        lstm_in = e.reshape(batch_size, n_channels*n_freq, n_frames).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_freq, n_frames)
        
        idx = len(e_list)
        d = lstm_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1            
            d = layer(_padded_cat(d, e_list[idx]))
            if self.debug:
                print(f"decoder_{i}: {d.shape}")
        
        return d

class LearnableSigmoid_1d(nn.Module):
      def __init__(self, in_features, beta=1):
            super().__init__()
            self.beta = beta
            self.slope = nn.Parameter(torch.ones(in_features))
            self.slope.requiresGrad = True

      def forward(self, x):
            return self.beta * torch.sigmoid(self.slope * x)
      
class MetricDisc(nn.Module):
      def __init__(self, dim=16, in_channel=2):
            super(MetricDisc, self).__init__()
            self.layers = nn.Sequential(
                  nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
                  nn.InstanceNorm2d(dim, affine=True),
                  nn.PReLU(dim),
                  nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
                  nn.InstanceNorm2d(dim*2, affine=True),
                  nn.PReLU(dim*2),
                  nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
                  nn.InstanceNorm2d(dim*4, affine=True),
                  nn.PReLU(dim*4),
                  nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
                  nn.InstanceNorm2d(dim*8, affine=True),
                  nn.PReLU(dim*8),
                  nn.AdaptiveMaxPool2d(1),
                  nn.Flatten(),
                  nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
                  nn.Dropout(0.3),
                  nn.PReLU(dim*4),
                  nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
                  LearnableSigmoid_1d(1)
            )

      def forward(self, x, y):
            xy = torch.cat([x, y], dim=1)
            return self.layers(xy)

class SingleRNN(torch.nn.Module):
    """Module for a RNN block.
    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(torch.nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output

class LifterModel(torch.nn.Module):
      def __init__(self, in_size=257, hidden_size=128, n_layers=2, rnn_type="LSTM", 
                 bidirectional=False, **kwargs):
            super().__init__()
            kwargs = {
                  "rnn_type": rnn_type,
                  "hidden_size": hidden_size,
                  "n_layers": 1,
                  "dropout": 0,
                  "bidirectional": bidirectional,
            }
            first_rnn = SingleRNN(input_size=in_size, **kwargs)
            self.rnns = torch.nn.ModuleList([first_rnn])
            for _ in range(n_layers - 1):
                  self.rnns.append(SingleRNN(input_size=first_rnn.output_size, **kwargs))
                  
            self.linear = torch.nn.Linear(self.rnns[-1].output_size, in_size)

      def forward(self, x):
            """Input shape: [batch, ..., time]"""
            # Permute to [batch, time, ...]
            x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
            # RNN + Linear expect [batch, time, rest]
            permuted_shape = x.shape
            x = x.reshape(*x.shape[:2], -1)
            for rnn in self.rnns:
                  x = rnn(x)
            x = self.linear(x).reshape(*permuted_shape)
            # Permute back to [batch, ..., time]
            x = x.permute(0, *range(2, x.ndim), 1)
            return x