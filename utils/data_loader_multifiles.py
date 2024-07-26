# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import netCDF4 as nc
import torch.distributed as dist
import xarray as xr

ORO_FILE = "/data_aip05/gsaliou/era5/global/oro/geopotential_2.5_global.nc"

def get_data_loader(params, data, distributed, train):
    #dataset = GetDataset(params, files_pattern, train)
    ds = xr.load_dataarray(data)
    ds = ds[: , :, :, 0:72] # Keep only the first 72 latitudes
    dataset = InMemoryDataset(ds)
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=params.batch_size,
                            num_workers=1,
                            shuffle=(not distributed and train),
                            sampler=sampler if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

def generate_pe(shape):
    """ This is a function that stack the positionnal encoding of the file"""
    
    img_shape_x, img_shape_y = shape
    
    x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
    x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
    y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
    y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
    
    grid_x1, grid_y1 = np.meshgrid(y1, x1)
    grid_x2, grid_y2 = np.meshgrid(y2, x2)
    return np.stack((grid_x1, grid_y1, grid_x2, grid_y2))
    
class InMemoryDataset(Dataset):
    def __init__(self, ds):
        self.data = torch.from_numpy(ds.values)
        self.coords = ds.coords
        self.oro = torch.from_numpy(xr.load_dataarray(ORO_FILE).values).squeeze()
        self.oro = self.oro[: , 0:72] # Keep only the first 72 latitudes
        self.oro = (self.oro - self.oro.mean()) / self.oro.std()
        self.pe = torch.from_numpy(generate_pe(self.oro.shape))
    
    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.concat([x, self.oro[None], self.pe])

        y = self.data[idx+1]
        return x,y