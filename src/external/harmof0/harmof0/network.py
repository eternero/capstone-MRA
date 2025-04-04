from math import sin
import torch
from torch._C import has_openmp
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .layers import MRDConv, FRDConv, WaveformToLogSpecgram

def dila_conv_block(
    in_channel, out_channel,
    bins_per_octave,
    n_har,
    dilation_mode,
    dilation_rate,
    dil_kernel_size,
    kernel_size,        # Removed defaults from these two...
    padding,            # no reason to have them as far as I know.
):

    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channel)

    # CREATES AN MRDC-CONV BLOCK
    if(dilation_mode == 'log_scale'):
        a = np.log(np.arange(1, n_har + 1))/np.log(2**(1.0/bins_per_octave))
        dilation_list = a.round().astype(np.int64)
        conv_log_dil = MRDConv(out_channel, out_channel, dilation_list)
        return nn.Sequential(
            conv,nn.ReLU(),
            conv_log_dil,nn.ReLU(),
            batch_norm,
            # pool
        )

    # CREATES AN FRDC-CONV BLOCK
    elif(dilation_mode == 'fixed_causal'):
        dilation_list = np.array([i * dil_kernel_size[1] for i in range(dil_kernel_size[1])])
        causal_conv = FRDConv(out_channel, out_channel, dil_kernel_size, dilation=[1, dilation_rate])
        return nn.Sequential(
            conv,nn.ReLU(),
            causal_conv,nn.ReLU(),
            batch_norm,
            # pool
        )

    # SUPPOSED TO CREATE AN SD-CONV BLOCK, BUT WHATEVER.
    elif(dilation_mode == 'fixed'):
        conv_dil = nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size, padding=[0, dilation_rate], dilation=[1, dilation_rate])

        return nn.Sequential(
            conv,nn.ReLU(),
            conv_dil,nn.ReLU(),
            batch_norm,
            # pool
        )
    else:
        assert False, "unknown dilation type: " + dilation_mode


class HarmoF0(nn.Module):
    def __init__(self,
            sample_rate=16000,
            n_freq=512,
            n_har=12,
            bins_per_octave=12 * 4,
            dilation_modes=['log_scale', 'fixed', 'fixed', 'fixed'],
            dilation_rates=[48, 48, 48, 48],
            logspecgram_type='logharmgram',
            channels=[32, 64, 128, 128],
            fmin=27.5,
            freq_bins=88 * 4,
            dil_kernel_sizes= [[1, 3], [1,3], [1,3], [1,3]],
        ):
        super().__init__()


        self.freq_bins = freq_bins
        self.n_freq    = n_freq
        n_fft          = n_freq * 2


        self.logspecgram_type        = logspecgram_type
        self.waveform_to_logspecgram = WaveformToLogSpecgram(sample_rate, n_fft, fmin, bins_per_octave, freq_bins, n_freq, logspecgram_type) #, device


        """
        In this segment below, I've decided to delete the old code and re-write in a more readable way.
        After all, the current code uses hardcoding either way, but it is simply fucking unreadable.
        For some clarification, check out page no.4 in the HarmoF0 research paper, which provides
        the hyperparameters of convolution layers.
        """

        # Initialize the MRDC-Conv Block
        self.block_1 = dila_conv_block(in_channel      = 1,             # 1  input
                                       out_channel     = 32,            # 32 outputs in this layer.
                                       bins_per_octave = 48,            # Recall that they use Q = 48.
                                       n_har           = 12,            # Also stated in the paper, sec.3.1.2
                                       dilation_mode   = 'log_scale',   # Log Scle is just MRDC-Conv
                                       dilation_rate   = 48,            # NOTE : This requires refactoring in `dila_conv_block`... the way they handle this code is terrible.
                                       dil_kernel_size = [1,3],         # NOTE : This doesn't make any sense either. The paper has the kernel size as 11x1. However the previous code had them all at [1,3]
                                       kernel_size     = [3,3],         # This does match up with the paper I guess. The convolutions always have a kernel size of 3x3.
                                       padding         = [1,1],         # No idea, but it's always at 1x1 so I leave it as is.
                                       )

        # NOTE : At this point, the bins are for some reason divided by 2...
        # Initialize the first of three SD-Conv Blocks.
        self.block_2 = dila_conv_block(in_channel      = 32,            # 32 inputs from prevous layer
                                       out_channel     = 64,            # 64 outputs in this layer.
                                       bins_per_octave = 24,            # NOTE : As mentioned, this was divided by 2 atp... Should check why and try it out without the division.
                                       n_har           = 3,             # NOTE : Strangely enough, after the first layer, all other ones have `n_har = 3`, not sure if this is correct.
                                       dilation_mode   = 'fixed',       # NOTE : Not sure why these are 'fixed' which is just Conv2D... They're supposed to be Standard Dilated Convolutions, not sure if this does it.
                                       dilation_rate   = 48,
                                       dil_kernel_size = [1,3],
                                       kernel_size     = [3,3],
                                       padding         = [1,1],
                                       )

        # Initialize the second of three SD-Conv Blocks.
        self.block_3 = dila_conv_block(in_channel      = 64,            # 64  inputs from prevous layer
                                       out_channel     = 128,           # 128 outputs in this layer.
                                       bins_per_octave = 24,
                                       n_har           = 3,
                                       dilation_mode   = 'fixed',
                                       dilation_rate   = 48,
                                       dil_kernel_size = [1,3],
                                       kernel_size     = [3,3],
                                       padding         = [1,1],
                                       )

        # Initialize the third of three SD-Conv Blocks.
        self.block_4 = dila_conv_block(in_channel      = 128,           # 128 inputs from prevous layer
                                       out_channel     = 128,           # 128 outputs in this layer.
                                       bins_per_octave = 24,
                                       n_har           = 3,
                                       dilation_mode   = 'fixed',
                                       dilation_rate   = 48,
                                       dil_kernel_size = [1,3],
                                       kernel_size     = [3,3],
                                       padding         = [1,1],
                                       )

        # Initialize the two remaining convolutional layers
        self.conv_5  = nn.Conv2d(in_channels = 128,
                                 out_channels= 64,
                                 kernel_size = [1,1]
                                )

        self.conv_6  = nn.Conv2d(in_channels = 64,
                                 out_channels= 1,
                                 kernel_size = [1,1]
                                )


    def forward(self, waveforms):
        # input: [b x num_frames x frame_len]
        # output: [b x num_frames x 352], [b x num_frames x 352]

        specgram = self.waveform_to_logspecgram(waveforms).float()
        # => [b x 1 x num_frames x n_bins]
        x = specgram[None, :]

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        # [b x 128 x T x 352] => [b x 64 x T x 352]
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x, dim=1)
        # x = torch.clip(x, 1e-4, 1 - 1e-4)
        # => [num_frames x n_bins]
        return x, specgram


