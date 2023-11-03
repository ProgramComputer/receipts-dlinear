
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Modified https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py 
#DModel
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # get average

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1) # repeats along dims
        x = torch.cat([front, x, end], dim=1) # concatenates front ,x ,end
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean # res is trend
class DModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self,config):
        super(DModel, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Decomposition Kernel Size
        kernel_size = config.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.channels = 1

  
        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
      
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init,trend_init: [Batch, Channel, Input length]
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
