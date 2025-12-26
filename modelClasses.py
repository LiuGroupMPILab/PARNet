# import libraries
from cmath import inf
import torch
from torch import nn
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F
from reconUtils import *
import os
import matplotlib.pyplot as plt
trainedNetOffset = "./"
    
# pp-mpi denoisers:

class dense_block(nn.Module):
    def __init__(self, in_channels, addition_channels, bias = True):
        super(dense_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1, bias = bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdb(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense, bias = True):
        super(rdb, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_block(in_channels+i*growth_at_each_dense,growth_at_each_dense, bias = bias))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv2d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0, bias = bias)
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))
class rdnDenoiserResRelu(nn.Module):#RDN残差密集块
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnDenoiserResRelu,self).__init__()
        
        self.conv0 = nn.Conv2d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias)
        self.conv3 = nn.Conv2d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1, bias = bias)
        self.conv4 = nn.Conv2d(in_channels=nb_of_features, out_channels= out_channel, kernel_size=3,stride=1,padding=1, bias = bias)
        self.lastReLU = nn.ReLU(inplace=False)
        # self.seblock = SEBlock(nb_of_features)
    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        residual1 = x
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)

        x = self.conv3(x) +residual0
        return self.lastReLU(self.conv4(x) + x_init)

class FBPDenoiserResRelu(nn.Module):  # RDN残差密集块
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel,
                 bias=True):
        super(FBPDenoiserResRelu, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.rdbs = nn.ModuleList()
        self.nanets = nn.ModuleList()
        # for i in range(0, 2):
        #     self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias=bias))
        for i in range(0, 1):
            self.nanets.append(NAFBlock(c=nb_of_features))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks * nb_of_features, out_channels=nb_of_features, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.conv5 = nn.Conv2d(in_channels=int(nb_of_features), out_channels=nb_of_features, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=int(nb_of_features), out_channels=nb_of_features, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=nb_of_features, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.lastReLU = nn.ReLU(inplace=False)

        # self.seblock = SEBlock(nb_of_features)

    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        residual1 = x
        rdb_outs = list()
        nan_outs = list()
        y = x
        # for layer in self.rdbs:
        #     x = layer(x)
        #     rdb_outs.append(x)
        # x = torch.cat(rdb_outs, dim=1)
        # x = self.conv2(x)
        for layer in self.nanets:
            y = layer(y)
            nan_outs.append(y)
        y = torch.cat(nan_outs, dim=1)
        y = self.conv5(y)
        # x = torch.cat([x, y], dim=1)

        x = self.conv3(y) + residual0
        return self.lastReLU(self.conv4(x) + x_init)
pair = lambda x: x if isinstance(x, tuple) else (x, x)
# class rdnnafDenoiserResRelu(nn.Module):  # RDN残差密集块
#     def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel,
#                  bias=True):
#         super(rdnnafDenoiserResRelu, self).__init__()
#
#         self.conv0 = nn.Conv2d(input_channels, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.rdbs = nn.ModuleList()
#         self.nanets = nn.ModuleList()
#         for i in range(0, nb_of_blocks):
#             self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias=bias))
#         # for i in range(0, 4):
#         #     self.nanets.append(NAFBlock(c=nb_of_features))
#         self.conv2 = nn.Conv2d(in_channels=nb_of_blocks * nb_of_features, out_channels=nb_of_features, kernel_size=1,
#                                stride=1, padding=0, bias=bias)
#         # self.conv5 = nn.Conv2d(in_channels=int(1* nb_of_blocks * nb_of_features), out_channels=nb_of_features, kernel_size=1,
#         #                        stride=1, padding=0, bias=bias)
#         self.conv3 = nn.Conv2d(in_channels=int(1 * nb_of_features), out_channels=nb_of_features, kernel_size=3, stride=1,
#                                padding=1, bias=bias)
#         self.conv4 = nn.Conv2d(in_channels=nb_of_features, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
#                                bias=bias)
#         self.lastReLU = nn.ReLU(inplace=False)
#
#         # self.seblock = SEBlock(nb_of_features)
#
#     def forward(self, x):
#         x_init = x
#         x = self.conv0(x)
#         residual0 = x
#         x = self.conv1(x)
#         residual1 = x
#         rdb_outs = list()
#         y = x
#         for layer in self.rdbs:
#             x = layer(x)
#             rdb_outs.append(x)
#         x = torch.cat(rdb_outs, dim=1)
#         x = self.conv2(x)
#         # for layer in self.nanets:
#         #     y = layer(y)
#         #     nan_outs.append(y)
#         # y = torch.cat(nan_outs, dim=1)
#         # y = self.conv5(y)
#         # x = torch.cat([x, y], dim=1)
#
#         x = self.conv3(x) + residual0
#         return self.lastReLU(self.conv4(x) + x_init)

# 有一点提升的结构
class rdnnafDenoiserResRelu(nn.Module):  # RDN残差密集块
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel,
                 bias=True):
        super(rdnnafDenoiserResRelu, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.rdbs = nn.ModuleList()
        self.nanets = nn.ModuleList()
        for i in range(0, 2):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias=bias))
        for i in range(0, 4):
            self.nanets.append(NAFBlock(c=nb_of_features))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks * nb_of_features, out_channels=nb_of_features, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.conv5 = nn.Conv2d(in_channels=int(2 * nb_of_features), out_channels=nb_of_features, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=int(2*nb_of_features), out_channels=nb_of_features, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=nb_of_features, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.lastReLU = nn.ReLU(inplace=False)

        # self.seblock = SEBlock(nb_of_features)

    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        residual1 = x
        rdb_outs = list()
        nan_outs = list()
        y = x
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        for layer in self.nanets:
            y = layer(y)
            nan_outs.append(y)
        y = torch.cat(nan_outs, dim=1)
        y = self.conv5(y)
        x = torch.cat([x, y], dim=1)

        x = self.conv3(x) + residual0
        return self.lastReLU(self.conv4(x) + x_init)
pair = lambda x: x if isinstance(x, tuple) else (x, x)

# pp-mpi denoiser with spectral normalization:


# from torch.nn.utils.parametrizations import spectral_norm

class consistencyNetworkMDspectral(nn.Module):
    def __init__(self, nb_of_features, nb_of_blocks, useNormalization = 0, numDim = 2, bias = True):
        super(consistencyNetworkMDspectral,self).__init__()

        self.numDim = numDim

        if numDim == 2:
            theConv = nn.Conv2d
        elif numDim == 1:
            theConv = nn.Conv1d
        if numDim == 3:
            theConv = nn.Conv3d
        
        self.conv0 = spectral_norm(theConv(4,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias))
        self.relu0 = nn.ReLU()
        self.convs = nn.ModuleList()
        for _ in range(nb_of_blocks):
            self.convs.append(spectral_norm(theConv(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)))
            self.convs.append(nn.ReLU())

        self.midConvs = nn.Sequential(*self.convs)

        self.convL = spectral_norm(theConv(in_channels=nb_of_features, out_channels= 2, kernel_size=3,stride=1,padding=1, bias = bias))

        self.useNormalization = useNormalization

        self.preProcessCls = usePreProcessedSelElems()
        # self.reluL = nn.ReLU()
        self.numAngle = 60 # num simulation angles

    def forward(self, s, y, epsilon, Ua = 1):
        if y.shape[2] > 2e3: # experimental data
            if isinstance(Ua, torch.Tensor):
                z2Ext = self.preProcessCls.projectA(F.linear(s, Ua)) # extended s
                yExt = self.preProcessCls.projectA(F.linear(y, Ua)) # extended Y
            else:
                z2Ext = self.preProcessCls.projectA(s) # extended s
                yExt = self.preProcessCls.projectA(y) # extended Y
            yShape = yExt.shape
            if self.numDim == 3:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], yShape[2], 10, 13)
                yExt = yExt.reshape(yShape[0], yShape[1], yShape[2], 10, 13)
            elif self.numDim == 1:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], yShape[2] * 10 * 13)
                yExt = yExt.reshape(yShape[0], yShape[1], yShape[2] * 10 * 13)
        else: # simulated data
            yExt = y.reshape(y.shape[0], 2, -1, self.numAngle)
            z2Ext = s.reshape(s.shape[0], 2, -1, self.numAngle)
            yShape = yExt.shape
            if self.numDim == 1:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], -1)
                yExt = yExt.reshape(yShape[0], yShape[1], -1)
            
        net1 = self.relu0(self.conv0(torch.concat((yExt, z2Ext), dim = 1)))

        net2 = self.midConvs(net1)
        netOut = self.convL(net2).reshape(yShape)

        if y.shape[2] > 2e3: # experimental data
            if isinstance(Ua, torch.Tensor):
                netUnProj = F.linear(self.preProcessCls.unProjectA(netOut), Ua.T)
            else:
                netUnProj = self.preProcessCls.unProjectA(netOut)
        else:
            netUnProj = netOut.reshape(y.shape)

        if self.useNormalization > 0:
            # nrmVal = torch.linalg.norm(s - y, dim=(1,2))
            if self.useNormalization == 1:
                nrmVal = torch.linalg.norm(netUnProj - y, dim=(1,2)) # limit innovation rate
            elif self.useNormalization == 2:
                nrmVal = torch.linalg.norm(netUnProj - 0, dim=(1,2)) # limit innovation rate
            elif self.useNormalization == 3:
                return netUnProj
            elif self.useNormalization == 4:
                return proj2Tmtx(netUnProj, y, epsilon) # return reference Value back

            if isinstance(epsilon, torch.Tensor):
                theInd = nrmVal < epsilon.squeeze()
                nrmVal[theInd] = epsilon.squeeze()[theInd]
            else:
                nrmVal[nrmVal < epsilon] = epsilon
            netUnProj *= epsilon / nrmVal[:,None,None]

        # netOut = (epsilon / nrmVal[:,None,None]) * (z2Ext - yExt)

        # return y + F.linear(self.preProcessCls.unProjectA(yExt + netOut), Ua.T) # return reference Value back
        return y + netUnProj # return reference Value back

class dense_blockSpectral(nn.Module):
    def __init__(self, in_channels, addition_channels, bias = True):
        super(dense_blockSpectral, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1, bias = bias))
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdbSpectral(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense, bias = True):
        super(rdbSpectral, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_blockSpectral(in_channels+i*growth_at_each_dense,growth_at_each_dense, bias = bias))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = spectral_norm(nn.Conv2d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0, bias = bias))
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))

class rdnDenoiserResReluSpectral(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnDenoiserResReluSpectral,self).__init__()
        
        self.conv0 = spectral_norm(nn.Conv2d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias))
        self.conv1 = spectral_norm(nn.Conv2d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias))
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdbSpectral(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias))
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1, bias = bias))
        self.conv4 = spectral_norm(nn.Conv2d(in_channels=nb_of_features, out_channels= out_channel, kernel_size=3,stride=1,padding=1, bias = bias))
        self.lastReLU = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) +residual0
        return self.lastReLU(self.conv4(x) + x_init)



class rdnStepEnd2End(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnStepEnd2End,self).__init__()
        self.conv0 = nn.Conv2d(in_channels=input_channels, out_channels=nb_of_features, kernel_size=3,stride=1,padding=1, bias = bias)
        
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv1 = nn.Conv2d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias)
        self.conv2 = nn.Conv2d(in_channels=nb_of_features, out_channels= input_channels,kernel_size=3,stride=1,padding=1, bias = bias)
        
    def forward(self,x):
        x = self.conv0(x)
        residual0 = x
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv1(x) +residual0
        return self.conv2(x)   

class dataConsistencyBlockWithMtxA(nn.Module):
    def __init__(self, M):
        super(dataConsistencyBlockWithMtxA,self).__init__()
        self.M = M

    def forward(self, x, x_lsqr):
        return F.linear(x.reshape(x.shape[0], 1, -1), self.M).reshape(x_lsqr.shape) + x_lsqr

    
class rdn2End(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias = True):
        super(rdn2End,self).__init__()
        
        self.sharedNet = rdnStepEnd2End(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
       

    def forward(self, x, x_lsConst, M):
        
        return self.sharedNet(x)


class rdnADMMnet(nn.Module):
    def __init__(self,input_channels, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias = True, numDim = 2):
        super(rdnADMMnet,self).__init__()
        
        if numDim == 2:
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                        layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        self.nb_of_steps = nb_of_steps

    def forward(self, datatC, AtC, x_in = None, MtC = 1, epsilon = 0):
        Nbatch = datatC.shape[0]
        x_inShape = x_in.shape
        
        x_conv = x_in.reshape(Nbatch, -1)
        z0 = x_in.reshape(x_conv.shape)
        z2 = F.linear(z0.reshape(Nbatch, -1), AtC)
        
        d0 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC.squeeze())
        
        for _ in range(self.nb_of_steps):
            r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
            x = F.linear(r, MtC)
            Ax = F.linear(x, AtC)
            
            z0 = self.sharedNet((x - d0).reshape(x_inShape)).reshape(Nbatch, -1) 
            
            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2
                
        return x.reshape(x_inShape)

class usePreProcessedSelElems():
    def __init__(self, myLoadedFile = "./inhouseData/selElems.mat"):
        self.myLoadedFile = myLoadedFile
        self.selElems = loadmat(self.myLoadedFile)["selElems"].astype("bool").squeeze()
        # self.selElems = torch.zeros(14820, dtype=torch.bool)
        # self.selElems[:4040] = True
        self.numAngle = 60
        self.numFreq = 247
        self.numFilt = 13 * 10


    def projectA(self,y):
        halfSize = y.shape[2] // 2
        dnm = torch.zeros((y.shape[0], 2, self.numAngle * self.numFreq), dtype=y.dtype, device = y.device).type_as(y)
        self.useElem = self.selElems
        dnm[:, :, self.useElem] = y.reshape(y.shape[0], 2, halfSize)

        return dnm.reshape(y.shape[0], 2, self.numAngle, self.numFreq)[:, :, :, :self.numFilt]

    def unProjectA(self,y):
        Nbatch = y.shape[0]

        dnm = torch.zeros((y.shape[0], 2, self.numAngle, self.numFreq), dtype=y.dtype, device = y.device).type_as(y)

        dnm[:, :, :, :self.numFilt] = y
        dnm = dnm.reshape(y.shape[0], 2, self.numAngle * self.numFreq)[:, :, self.useElem]

        yOut = dnm.reshape(Nbatch, 1, -1)

        # yOut = torch.cat((dnm[:, 0, :], dnm[:, 1, :]), dim = 2)

        return yOut

class consistencyNetworkMD(nn.Module):#LC学习一致性块
    def __init__(self, nb_of_features, nb_of_blocks, useNormalization = 0, numDim = 2, bias = True):
        super(consistencyNetworkMD,self).__init__()

        self.numDim = numDim

        if numDim == 2:
            theConv = nn.Conv2d
        elif numDim == 1:
            theConv = nn.Conv1d
        if numDim == 3:
            theConv = nn.Conv3d
        
        self.conv0 = theConv(4,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.relu0 = nn.ReLU()
        self.convs = nn.ModuleList()
        for _ in range(nb_of_blocks):
            self.convs.append(theConv(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias))
            self.convs.append(nn.ReLU())

        self.midConvs = nn.Sequential(*self.convs)

        self.convL = theConv(in_channels=nb_of_features, out_channels= 2, kernel_size=3,stride=1,padding=1, bias = bias)

        self.useNormalization = useNormalization# % 3
        # if useNormalization > 3:
        #     self.residual = False
        # else:
        #     self.residual = True

        self.preProcessCls = usePreProcessedSelElems()
        # self.reluL = nn.ReLU()
        self.numAngle = 60 # num simulation angles

    def forward(self, s, y, epsilon, Ua = 1):
        if y.shape[2] > 2e3: # experimental data
            if isinstance(Ua, torch.Tensor):
                z2Ext = self.preProcessCls.projectA(F.linear(s, Ua)) # extended s
                yExt = self.preProcessCls.projectA(F.linear(y, Ua)) # extended Y
            else:
                z2Ext = self.preProcessCls.projectA(s) # extended s
                yExt = self.preProcessCls.projectA(y) # extended Y
            yShape = yExt.shape
            if self.numDim == 3:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], yShape[2], 10, 13)
                yExt = yExt.reshape(yShape[0], yShape[1], yShape[2], 10, 13)
            elif self.numDim == 1:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], yShape[2] * 10 * 13)
                yExt = yExt.reshape(yShape[0], yShape[1], yShape[2] * 10 * 13)
        else: # simulated data
            yExt = y.reshape(y.shape[0], 2, -1, self.numAngle)
            z2Ext = s.reshape(s.shape[0], 2, -1, self.numAngle)
            yShape = yExt.shape
            if self.numDim == 1:
                z2Ext = z2Ext.reshape(yShape[0], yShape[1], -1)
                yExt = yExt.reshape(yShape[0], yShape[1], -1)
            
        net1 = self.relu0(self.conv0(torch.concat((yExt, z2Ext), dim = 1)))

        net2 = self.midConvs(net1)
        netOut = self.convL(net2).reshape(yShape)

        if y.shape[2] > 2e3: # experimental data
            if isinstance(Ua, torch.Tensor):
                netUnProj = F.linear(self.preProcessCls.unProjectA(netOut), Ua.T)
            else:
                netUnProj = self.preProcessCls.unProjectA(netOut)
        else:
            netUnProj = netOut.reshape(y.shape)

        if self.useNormalization > 0:
            # nrmVal = torch.linalg.norm(s - y, dim=(1,2))
            if self.useNormalization == 1:#training there
                nrmVal = torch.linalg.norm(netUnProj - y, dim=(1,2)) # limit innovation rate
            elif self.useNormalization == 2:
                nrmVal = torch.linalg.norm(netUnProj - 0, dim=(1,2)) # limit innovation rate
            elif self.useNormalization == 3:
                return netUnProj
            elif self.useNormalization == 4:
                return proj2Tmtx(netUnProj, y, epsilon) # return reference Value back

            if isinstance(epsilon, torch.Tensor):
                theInd = nrmVal < epsilon.squeeze()
                nrmVal[theInd] = epsilon.squeeze()[theInd]
            else:#there
                # nrmVal[nrmVal < epsilon] = epsilon
                nrmVal = torch.where(nrmVal < epsilon, torch.tensor(epsilon, device=nrmVal.device, dtype=nrmVal.dtype),nrmVal)

            # netUnProj *= epsilon / nrmVal[:,None,None]
            netUnProj = netUnProj * (epsilon / nrmVal[:, None, None])

        # netOut = (epsilon / nrmVal[:,None,None]) * (z2Ext - yExt)

        # return y + F.linear(self.preProcessCls.unProjectA(yExt + netOut), Ua.T) # return reference Value back
        return y + netUnProj # return reference Value back


class rdnADMMLDnet(nn.Module): # learned data consistency
    def __init__(self,input_channels, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL = 0, bias = True, numDim = 2, consistencyDim = 3):
        super(rdnADMMLDnet,self).__init__()
        
        if numDim == 2:
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                        layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        if consistencyDim == 0:
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL, numDim = consistencyDim).cuda()
            
        self.nb_of_steps = nb_of_steps
        
    def forward(self, datatC, AtC, Ua = 1, x_in = None, MtC = 1, epsilon = 0):
        Nbatch = datatC.shape[0]
        x_inShape = x_in.shape
        
        # if x_in is None:
        #     x_conv = F.linear(datatC, AtC.T)
        #     z0 = torch.zeros_like(x_conv, )
        #     z2 = torch.zeros_like(datatC)
        # else:
        x_conv = x_in.reshape(Nbatch, -1)
        z0 = x_in.reshape(x_conv.shape)
        z2 = F.linear(z0.reshape(Nbatch, -1), AtC)
        
        d0 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC.squeeze())
        
        for _ in range(self.nb_of_steps):
            r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
            x = F.linear(r, MtC)
            Ax = F.linear(x, AtC)

            z0 = self.sharedNet((x - d0).reshape(x_inShape)).reshape(Nbatch, -1)

            z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon, Ua).reshape(Nbatch, -1)

            # z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
            #             datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2
                
        return x.reshape(x_inShape)



class rdnLDFixedPtspc(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL = 0, bias = True, numDim = 2, consistencyDim = 3):

        super(rdnLDFixedPtspc,self).__init__()
        if numDim == 2:
            self.sharedNet = rdnDenoiserResReluSpectral(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        if consistencyDim == 0:
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMDspectral(nb_of_featuresL, nb_of_blocksL, useNormalizationL, numDim = consistencyDim).cuda()

        # self.nb_of_steps = nb_of_steps
        
    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams

        # tempTime = time.time()

        Nbatch = datatC.shape[0]
        # x_inShape[0] = Nbatch
        Nimg = np.prod(x_inShape[2:])
        # Ndata = datatC.shape[2] # (theIn.shape[1] - 2 * Nimg)
        x_in, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:]
        
        # x_inShape = x_in.shape
        
        # tempTime2 = time.time()
        Ax = F.linear(x_in, AtC)
        # print("Ax time: ",time.time() - tempTime2)

        # tempTime2 = time.time()
        z0 = self.sharedNet((x_in - d0).reshape(x_inShape)).reshape(Nbatch, -1) 
        # print("Network time: ",time.time() - tempTime2)
        
        # z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
        #             datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)
        z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1),
                    datatC.reshape(Nbatch, 1, -1), epsilon, Ua = 1).reshape(Nbatch, -1)

        r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
        x = F.linear(r, MtC)
        Ax = F.linear(x, AtC)
        
        d0 = d0 - x + z0
        d2 = d2 - Ax + z2
        # print("pass time: ",time.time() - tempTime)

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim = 1 )

import torch

def computeRowEnergy(A):
    """
    Calculate the energy of the system matrix rows
    """
    M = A.shape[0]
    energy = torch.zeros(M, dtype=torch.double)

    for m in range(M):
        energy[m] = torch.norm(A[m, :])
    return energy

def Kaczmarz(A, b, iterations=10, lambd=0, enforceReal=False, enforcePositive=True, shuffle=True):
    M = A.shape[0]
    N = A.shape[1]

    x = torch.zeros(N, dtype=b.dtype)
    residual = torch.zeros(M, dtype=x.dtype)

    energy = computeRowEnergy(A)

    rowIndexCycle = torch.arange(0, M)

    if shuffle:
        rowIndexCycle = rowIndexCycle[torch.randperm(M)]

    lambdIter = lambd * (torch.sum(energy * energy) / N)

    for l in range(iterations):
        for m in range(M):
            k = rowIndexCycle[m]
            if energy[k] > 0:
                beta = (b[k] - torch.dot(A[k, :], x) - torch.sqrt(lambdIter) * residual[k]) / (energy[k] ** 2 + lambdIter)

                x += beta * A[k, :].conj()

                residual[k] += torch.sqrt(lambdIter) * beta

        if enforceReal and x.is_complex():
            x.imag = 0
        if enforcePositive:
            x = x * (x.real > 0)

    return x
def Kaczmarz_NN(A, b, iterations=10, lambd=0, enforceReal=False, enforcePositive=True, shuffle=True):
    M = A.shape[0]
    N = A.shape[1]

    x = torch.zeros(N, dtype=b.dtype)
    residual = torch.zeros(M, dtype=x.dtype)

    energy = computeRowEnergy(A)

    rowIndexCycle = torch.arange(0, M)

    if shuffle:
        rowIndexCycle = rowIndexCycle[torch.randperm(M)]

    lambdIter = lambd * (torch.sum(energy * energy) / N)

    for l in range(iterations):
        for m in range(M):
            k = rowIndexCycle[m]
            if energy[k] > 0:
                beta = (b[k] - torch.dot(A[k, :], x) - torch.sqrt(lambdIter) * residual[k]) / (energy[k] ** 2 + lambdIter)

                x = x + beta * A[k, :].conj()

                residual[k] = residual[k] + torch.sqrt(lambdIter) * beta

        if enforceReal and x.is_complex():
            x.imag = 0
        if enforcePositive:
            x = x * (x.real > 0)

    return x
class kaczmarzrdnixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias=True, numDim=2):

        super(kaczmarzrdnixedPt, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features,
                                                  nb_of_blocks=nb_of_blocks,
                                                  layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                  out_channel=input_channels, bias=bias)


    def forward(self, theIn, fixedParams):
        b, A, energy, lambd, x_inShape = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()
        Nbatch = b.shape[0]

        Nimg = np.prod(x_inShape[2:])
        M = A.shape[0]

        lambdIter = lambd * (torch.sum(energy * energy) / Nimg)
        b = b.reshape(Nbatch, -1)  # (batch_size,M)
        x, residual = theIn[:, :Nimg], theIn[:, Nimg:]
        residual = residual.clone()  # 先克隆整个 residual，避免视图问题
        for k in range(5): #按行顺序求解
            Ak = A[k,:]  # (N,)

            bk = b[:, k]
            rk = residual[:, k]  # (batch_size,M)
            # 计算 beta，确保广播

            beta = (bk - torch.matmul(x, Ak)  - torch.sqrt(lambdIter) * rk) / (
                        energy[k] ** 2 + lambdIter)  # (batch_size,)

            # 更新 x 和 residual
            x = self.sharedNet((x+beta[:, None] * Ak.conj()).reshape(x_inShape)).reshape(Nbatch, -1)  # (batch_size, N)
            # x = x+beta[:, None] * Ak.conj()
            residual[:, k] = residual[:, k] + torch.sqrt(lambdIter) * beta
            # residual[:,k] = residual[:,k] + torch.sqrt(lambdIter) * beta

        return torch.cat((x.reshape(Nbatch, -1), residual.reshape(Nbatch, -1)), dim=1)
class ngcrrdnixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL, bias=True, numDim=2, consistencyDim=0):

        super(ngcrrdnixedPt, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features,
                                                  nb_of_blocks=nb_of_blocks,
                                                  layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                  out_channel=input_channels, bias=bias)
        if consistencyDim == 0:#LC学习一致性块
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL,
                                                       numDim=consistencyDim).cuda()

    def forward(self, theIn, fixedParams):
        b, S, lambda_reg, x_inShape, epsilon = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()
        Nbatch = b.shape[0]

        Nimg = np.prod(x_inShape[2:])
        M = S.shape[0]
        c, r, z, p = theIn[:, :Nimg], theIn[:, Nimg:Nimg+M], theIn[:, Nimg+M:Nimg+M+Nimg], theIn[:, Nimg+M+Nimg:Nimg+M+2*Nimg]
        for inter in range(3):
            v = torch.matmul(p, S.T)  # 计算搜索方向在系统矩阵下的投影
            alpha = torch.sum(z * z, dim=1) / (torch.sum(v * v, dim=1) + lambda_reg * torch.sum(p * p, dim=1))
            if inter >= 1:
                # c = self.sharedNet((c + alpha.unsqueeze(1) * p).reshape(x_inShape)).reshape(Nbatch, -1)

                a = c
                b = alpha.unsqueeze(1) * p
                # c = torch.cat((a, b), dim=1)

                # c = self.sharedNet(c.reshape([64, 2, 21, 21])).reshape(Nbatch, -1)
                a = self.sharedNet(a.reshape(x_inShape)).reshape(Nbatch, -1)
                b = self.sharedNet(b.reshape(x_inShape)).reshape(Nbatch, -1)

                c = a + b
            else:
                c = c + alpha.unsqueeze(1) * p  # 更新解

            r = r - alpha.unsqueeze(1) * v # 更新残差
            # r = self.consistencyNet((r - alpha.unsqueeze(1) * v).reshape(Nbatch, 1, -1),  # LC学习一致性块
            #                     r.reshape(Nbatch, 1, -1), epsilon, Ua=1).reshape(Nbatch, -1)
            z_new = torch.matmul(r, S.conj()) - lambda_reg * c  # 计算新的梯度方向
            beta = torch.sum(z_new * z_new, dim=1) / torch.sum(z * z, dim=1)  # 计算 beta 用于更新搜索方向
            p = z_new + beta.unsqueeze(1) * p  # 更新搜索方向
            # p = self.searchNet(torch.cat((z_new, p, beta.unsqueeze(1)), dim=1))
            z = z_new  # 更新梯度方向
        # c = torch.clamp(c, min=-1e6, max=1e6)

        return torch.cat((c.reshape(Nbatch, -1), r, z, p), dim=1)
class ngcrrdnDCixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias=True, numDim=2):

        super(ngcrrdnixedPt, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features,
                                                  nb_of_blocks=nb_of_blocks,
                                                  layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                  out_channel=input_channels, bias=bias)


    def forward(self, theIn, fixedParams):
        b, S, lambda_reg, x_inShape = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()
        Nbatch = b.shape[0]

        Nimg = np.prod(x_inShape[2:])
        M = S.shape[0]
        c, r, z, p = theIn[:, :Nimg], theIn[:, Nimg:Nimg+M], theIn[:, Nimg+M:Nimg+M+Nimg], theIn[:, Nimg+M+Nimg:Nimg+M+2*Nimg]
        for inter in range(5):
            v = torch.matmul(p, S.T)  # 计算搜索方向在系统矩阵下的投影
            alpha = torch.sum(z * z, dim=1) / (torch.sum(v * v, dim=1) + lambda_reg * torch.sum(p * p, dim=1))
            if inter >= 3:
                # c = self.sharedNet((c + alpha.unsqueeze(1) * p).reshape(x_inShape)).reshape(Nbatch, -1)

                a = c
                b = alpha.unsqueeze(1) * p
                # c = torch.cat((a, b), dim=1)

                # c = self.sharedNet(c.reshape([64, 2, 21, 21])).reshape(Nbatch, -1)
                a = self.sharedNet(a.reshape(x_inShape)).reshape(Nbatch, -1)
                b = self.sharedNet(b.reshape(x_inShape)).reshape(Nbatch, -1)

                c = a + b
            else:
                c = c + alpha.unsqueeze(1) * p  # 更新解

            r = r - alpha.unsqueeze(1) * v  # 更新残差
            z_new = torch.matmul(r, S.conj()) - lambda_reg * c  # 计算新的梯度方向
            beta = torch.sum(z_new * z_new, dim=1) / torch.sum(z * z, dim=1)  # 计算 beta 用于更新搜索方向
            p = z_new + beta.unsqueeze(1) * p  # 更新搜索方向
            # p = self.searchNet(torch.cat((z_new, p, beta.unsqueeze(1)), dim=1))
            z = z_new  # 更新梯度方向
        # c = torch.clamp(c, min=-1e6, max=1e6)

        return torch.cat((c.reshape(Nbatch, -1), r, z, p), dim=1)
class rdnLDFixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL = 0, bias = True, numDim = 2, consistencyDim = 3):

        super(rdnLDFixedPt,self).__init__()
        if numDim == 2:#RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        if consistencyDim == 0:#LC学习一致性块
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL, numDim = consistencyDim).cuda()
        
    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams#固定变量y，A，M，阈值
        # tempTime = time.time()

        Nbatch = datatC.shape[0]

        Nimg = np.prod(x_inShape[2:])

        x_in, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:]

        Ax = F.linear(x_in, AtC)

        z0 = self.sharedNet((x_in - d0).reshape(x_inShape)).reshape(Nbatch, -1) #RDN残差密集块
        # fnc1 = softTV(0.1 / 10, [21,21], 10)
        # z0 = fnc1(x_in - d0)
        z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1), #LC学习一致性块
                    datatC.reshape(Nbatch, 1, -1), epsilon, Ua = 1).reshape(Nbatch, -1)
        # z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
        #                datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)
        r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
        x = F.linear(r, MtC)
        Ax = F.linear(x, AtC)

        d0 = d0 - x + z0
        d2 = d2 - Ax + z2

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim = 1 )


class rdnLDFixedPt_(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL,
                 nb_of_blocksL, useNormalizationL=0, bias=True, numDim=2, consistencyDim=3):

        super(rdnLDFixedPt_, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features,
                                                  nb_of_blocks=nb_of_blocks,
                                                  layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                  out_channel=input_channels, bias=bias)

        if consistencyDim == 0:  # LC学习一致性块
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL,
                                                       numDim=consistencyDim).cuda()

    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()

        Nbatch = datatC.shape[0]

        Nimg = np.prod(x_inShape[2:])

        x, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2 * Nimg], theIn[:, 2 * Nimg:]
        for i in range(2):
            Ax = F.linear(x, AtC)

            z0 = self.sharedNet((x - d0).reshape(x_inShape)).reshape(Nbatch, -1)  # RDN残差密集块
            # fnc1 = softTV(0.1 / 10, [21,21], 10)
            # z0 = fnc1(x - d0)
            z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1), #LC学习一致性块
                        datatC.reshape(Nbatch, 1, -1), epsilon, Ua = 1).reshape(Nbatch, -1)
            # z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
            #                datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)
            r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
            x = F.linear(r, MtC)
            Ax = F.linear(x, AtC)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim=1)

class hybFixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL,
                 nb_of_blocksL, useNormalizationL=0, bias=True, numDim=2, consistencyDim=3):

        super(hybFixedPt, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features,
                                                  nb_of_blocks=nb_of_blocks,
                                                  layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                  out_channel=input_channels, bias=bias)

        if consistencyDim == 0:  # LC学习一致性块
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL,
                                                       numDim=consistencyDim).cuda()

    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()

        Nbatch = datatC.shape[0]

        Nimg = np.prod(x_inShape[2:])

        x, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2 * Nimg], theIn[:, 2 * Nimg:]
        d1 = d0
        for i in range(2):
            Ax = F.linear(x, AtC)

            # z0 = self.sharedNet((x - d0).reshape(x_inShape)).reshape(Nbatch, -1)  # RDN残差密集块
            fnc1 = softTV(0.1 / 10, [21,21], 10)
            z0 = fnc1(x - d0)
            # fnc2 = softTpos((1 - 0.1) / 10)
            # z1 = fnc2((x - d1))
            # z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1),  # LC学习一致性块
            #                          datatC.reshape(Nbatch, 1, -1), epsilon, Ua=1).reshape(Nbatch, -1)

            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                           datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

            r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
            x = F.linear(r, MtC)
            Ax = F.linear(x, AtC)

            d0 = d0 - x + z0
            # d1 = d1 - x + z1
            d2 = d2 - Ax + z2

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim=1)
class rdnADMMnetVerbose(nn.Module):
    def __init__(self,input_channels, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias = True):
        super(rdnADMMnetVerbose,self).__init__()
        
        self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        
        self.nb_of_steps = nb_of_steps
        
    def forward(self, datatC, AtC, x_in, MtC, epsilon):
        Nbatch = datatC.shape[0]
        x_inShape = x_in.shape
        
        # if x_in is None:
        #     x_conv = F.linear(datatC, AtC.T)
        #     z0 = torch.zeros_like(x_conv, )
        #     z2 = torch.zeros_like(datatC)
        # else:
        x_conv = x_in.reshape(Nbatch, -1)
        z0 = x_in.reshape(x_conv.shape)
        z2 = F.linear(z0.reshape(Nbatch, -1), AtC)
        
        d0 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC.squeeze())
        
        for _ in range(self.nb_of_steps):
            r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
            x = F.linear(r, MtC)
            Ax = F.linear(x, AtC)
            
            z0 = self.sharedNet((x - d0).reshape(x_inShape)).reshape(Nbatch, -1) 
            
            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2
            with torch.no_grad():
                print("z2: ", torch.norm(z2))
                print("z2Fark: ", torch.norm(z2 - datatC.reshape(Nbatch, -1)))
                
        return x.reshape(x_inShape)

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 2.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz = x0.shape[0]
    d = np.prod(x0.shape[1:])#[x,d0,d1]数据向量长度
        
    H = W = 1
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    # print(k)
    return X[:,k%m].view_as(x0), res

def fixedPointTekrarlar(f, x0, max_iter=50):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d = x0.shape
    H = W = 1
    out = x0.clone()
    for k in range(max_iter):
        out = f(out)
    # print(k)
    return out.view_as(x0), 0

def regularFixed(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d = x0.shape
    H = W = 1
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    # print(k)
    return X[:,k%m].view_as(x0), res

# import time
class rdnFixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias = True, numDim = 2):

        super(rdnFixedPt,self).__init__()
        if numDim == 2:
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        # self.nb_of_steps = nb_of_steps
        
    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams

        # tempTime = time.time()

        Nbatch = datatC.shape[0]
        # x_inShape[0] = Nbatch
        Nimg = np.prod(x_inShape[2:])
        # Ndata = datatC.shape[2] # (theIn.shape[1] - 2 * Nimg)
        x_in, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:]
        
        # x_inShape = x_in.shape
        
        # tempTime2 = time.time()
        Ax = F.linear(x_in, AtC)
        # print("Ax time: ",time.time() - tempTime2)

        # tempTime2 = time.time()
        z0 = self.sharedNet((x_in - d0).reshape(x_inShape)).reshape(Nbatch, -1) 
        # print("Network time: ",time.time() - tempTime2)
        
        z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                    datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

        r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
        x = F.linear(r, MtC)
        Ax = F.linear(x, AtC)
        
        d0 = d0 - x + z0
        d2 = d2 - Ax + z2
        # print("pass time: ",time.time() - tempTime)

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim = 1 )

        # return (x.reshape(x_inShape), Ax, d0, d2)

class rdnAcceleratedFixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias = True):

        super(rdnAcceleratedFixedPt,self).__init__()
        self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        # self.nb_of_steps = nb_of_steps

    def firstCallOutside(self, theIn, fixedParams):        
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams
        
        Nbatch = datatC.shape[0]
        Nimg = x_inShape[2] * x_inShape[3]
        Ndata = datatC.numel() // Nbatch
        z0Hat, d0Hat, z2Hat, d2Hat = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:2*Nimg + Ndata], theIn[:, 2*Nimg + Ndata:]
        self.z0 = z0Hat
        self.d0 = d0Hat
        self.z2 = z2Hat
        self.d2 = d2Hat
        self.costp = inf * torch.ones(Nbatch, ).float()
        self.alpha = torch.ones(Nbatch, ).float().cuda()
        
    def forward(self, theIn, fixedParams, firstCall = False):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams
        
        eta = 1

        Nbatch = datatC.shape[0]
        Nimg = x_inShape[2] * x_inShape[3]
        Ndata = datatC.numel() // Nbatch
        z0Hat, d0Hat, z2Hat, d2Hat = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:2*Nimg + Ndata], theIn[:, 2*Nimg + Ndata:]
        
        if firstCall:
            self.z0 = z0Hat
            self.d0 = d0Hat
            self.z2 = z2Hat
            self.d2 = d2Hat
            self.costp = inf * torch.ones(Nbatch, ).float()
            self.alpha = torch.ones(Nbatch, ).float().cuda()
            
        # x_inShape = x_in.shape
        
        r = (z0Hat + d0Hat) + F.linear(z2Hat + d2Hat, AtC.T)
        x = F.linear(r, MtC)
        Ax = F.linear(x, AtC)

        z0 = self.sharedNet((x - d0Hat).reshape(x_inShape)).reshape(Nbatch, -1) 
        
        z2 = proj2Tmtx((Ax - d2Hat).reshape(Nbatch, 1, -1),
                    datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)
        
        d0 = d0Hat - x + z0
        d2 = d2Hat - Ax + z2

        # with torch.no_grad(): # may work without the gradient, or with the other forward function
        cost = (torch.linalg.norm((d0 - d0Hat).reshape(Nbatch, -1), dim = (1))**2  + \
            torch.linalg.norm((d2 - d2Hat).reshape(Nbatch, -1), dim = (1))**2 + \
            1 * torch.linalg.norm((z0 - z0Hat).reshape(Nbatch, -1), dim = (1))**2 + \
            1 * torch.linalg.norm((z2 - z2Hat).reshape(Nbatch, -1), dim = (1))**2).cpu()

        alpha_new = (1 + (1 + 4 * self.alpha**2)**(1/2)) / 2
        tau = (self.alpha - 1) / alpha_new

        indList = cost > eta * self.costp
        alpha_new[indList] = 1
        tau[indList] = -1
        cost[indList] = 1/eta * self.costp[indList]

        z0Hat = z0 + tau[:, None] * (z0 - self.z0)
        d0Hat = d0 + tau[:, None] * (d0 - self.d0)
        z2Hat = z2 + tau[:, None] * (z2 - self.z2)
        d2Hat = d2 + tau[:, None] * (d2 - self.d2)
        self.alpha = alpha_new

        self.z0 = z0
        self.z2 = z2
        self.d0 = d0
        self.d2 = d2
        self.costp = cost

        return torch.cat((z0Hat.reshape(Nbatch, -1), d0Hat.reshape(Nbatch, -1), z2Hat.reshape(Nbatch, -1), d2Hat.reshape(Nbatch, -1)), dim = 1 )

import torch.autograd as autograd

class DEQFixedPoint(nn.Module):#paper中的主模型
    def __init__(self, f, solver, **kwargs): #f为RDN&LC子模型，solver是Anderson 加速法
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        # initialize state dict
        # self.f.sharedNet.load_state_dict(torch.load())
        
    def forward(self, initialVal):#theIn:[x0,d1,d0], theFixedPts
        # compute forward pass and re-engage autograd tape
        # with torch.no_grad():
        #     z, self.forward_res = self.solver(lambda z : self.f(z), initialVal, **self.kwargs)

        z = initialVal
        n = 1
        for i in range(n):
            z = self.f(z)
        
        # set up Jacobian vector product (without additional forward calls)
        # z0 = z.clone().detach().requires_grad_()
        # f0 = self.f(z0)
        # def backward_hook(grad):
        #     g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
        #                                        grad, **self.kwargs)
        #     return g
        #
        # if (z.requires_grad):
        #     z.register_hook(backward_hook)
        return z


class jhFixedPoint(nn.Module):  # paper中的主模型
    def __init__(self, f, solver, **kwargs):  # f为RDN&LC子模型，solver是Anderson 加速法
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        # initialize state dict
        # self.f.sharedNet.load_state_dict(torch.load())

    def forward(self, theIn, theFixedPts):  # theIn:[x0,d1,d0], theFixedPts
        # compute forward pass and re-engage autograd tape
        # print(initialVal.shape) torch.Size([64, 6162])
        # with torch.no_grad():
        #     z, self.forward_res = self.solver(lambda z: self.f(z, theFixedPts), theIn, **self.kwargs)

            # print(z.shape) torch.Size([64, 6162])
        z = theIn
        # print(initialVal.shape)
        z = self.f(z, theFixedPts)

        # set up Jacobian vector product (without additional forward calls)
        # z0 = z.clone().detach().requires_grad_()
        # f0 = self.f(z0, theFixedPts)
        #
        # def backward_hook(grad):
        #     g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
        #                                        grad, **self.kwargs)
        #     return g
        #
        # if (z.requires_grad):
        #     z.register_hook(backward_hook)
        return z
class DEQFixedPointE2E(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        # initialize state dict
        # self.f.sharedNet.load_state_dict(torch.load())
        
    def forward(self, initialVal, x, x2):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x, x2), initialVal, **self.kwargs)
        z = self.f(z, x, x2)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x, x2)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
        
        if (z.requires_grad):
            z.register_hook(backward_hook)
        return z

class rdnDenoiserResRelu3d(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnDenoiserResRelu3d,self).__init__()
        
        self.conv0 = nn.Conv3d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.conv1 = nn.Conv3d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb3d(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv2 = nn.Conv3d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias)
        self.conv3 = nn.Conv3d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1, bias = bias)
        self.conv4 = nn.Conv3d(in_channels=nb_of_features, out_channels= out_channel, kernel_size=3,stride=1,padding=1, bias = bias)
        self.lastReLU = nn.ReLU(inplace=False)
    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) +residual0
        return self.lastReLU(self.conv4(x) + x_init)
    
class dense_block3d(nn.Module):
    def __init__(self, in_channels, addition_channels, bias = True):
        super(dense_block3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1, bias = bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdb3d(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense, bias = True):
        super(rdb3d, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_block3d(in_channels+i*growth_at_each_dense,growth_at_each_dense, bias = bias))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv3d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0, bias = bias)
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))



def getModel(descriptor):
    """descriptor: lr_0.001_wd_1e-07_bs_256_mxNs_0.05_fixNs_0_data_400_bias_0_mnNs_0.0"""
#     rdnDenoiserNewNs_lr_0.001_wd_0.0_bs_64_mxNs_0.2_fixNs_1_data_mnNs_0_nF16_nB3_lieb5_gr8_mtx_4.5_3.5_svd_824
    descriptor = descriptor[:-11] if "\n_scheduled" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+tv" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+l1" in descriptor else descriptor
    folderPath = "./training/denoiser/"+descriptor
#     print(folderPath)
    fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
    filePath = folderPath + "/" +fileName
    
    if "rdnDenoiser" in descriptor:
        descriptor = descriptor[10:]
    nb_of_features = int(descriptor.split("_")[14][2:])
    nb_of_blocks = int(descriptor.split("_")[15][2:])
    layer_in_each_block = int(descriptor.split("_")[16][4:])
    growth_rate = int(descriptor.split("_")[17][2:])
    biasFlag = True

    model = rdnDenoiserResRelu(input_channels=1,
                nb_of_features=nb_of_features,
                nb_of_blocks=nb_of_blocks,
                layer_in_each_block=layer_in_each_block, 
                growth_rate=growth_rate,
                out_channel=1,
                bias = biasFlag).cuda()

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
#     print(sum(pVal.numel() for pVal in model.parameters() if pVal.requires_grad))
    print("num params of model: ", sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))
    return model


def getModelForRDN2E(descriptor, getMaxFlag = True):
    folderPath = trainedNetOffset + "training/rdn2e/"+descriptor
#     print(folderPath)

    fileName = "epoch150END.pth"
    if getMaxFlag:
        fayNeyn = [i for i in os.listdir(folderPath) if "max" in i]
        if len(fayNeyn) > 1:
            jj = -1
            for ffNames in fayNeyn:
                myVal = int(ffNames.split('max')[0][5:])
                if myVal > jj:
                    jj = myVal
            fayNeyn = ["epoch" + str(jj) + "max.pth"]
    else:
        fayNeyn = [i for i in os.listdir(folderPath) if "END" in i]

    # if os.path.exists(folderPath + "/" + fileName):
    if len(fayNeyn) > 0:
        fileName = fayNeyn[0]
        filePath = folderPath + "/" +fileName
    else:
        fayNeyns = os.listdir(folderPath)[0]
        folderPath += "/" + fayNeyns
        fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
        filePath = folderPath + "/" +fileName
    

    # filePath = folderPath + "/" +fileName
    if "rdnDenoiser" in descriptor:
        descriptor = descriptor[12:]
    
    splt = descriptor.split("_")

    print(splt)
    offset = 0
    if splt[1] == "ds":
        offset = 2
    
    nb_of_steps = int(splt[offset + 15])
    nb_of_features = int(splt[offset + 16][2:])
    nb_of_blocks = int(splt[offset + 17][2:])
    layer_in_each_block = int(splt[offset + 18][4:])
    growth_rate = int(splt[offset + 19][2:])
    lambdaVal = float(splt[offset + 20][3:])
    biasFlag = True

#     print(descriptor, filePath)
    model = rdn2End(
        1, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias=biasFlag)
    # print(filePath)

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
#     print(sum(pVal.numel() for pVal in model.parameters() if pVal.requires_grad))
    print("num params of model: ", sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))

    return model, lambdaVal

def getModel3dDenoiser(descriptor):
    'rdnDenoiserNewNs_lr_0.001_wd_0.0_bs_128_mxNs_0.05_fixNs_1_data_mnNs_0_nF16_nB4_lieb4_gr12_rMn1.0_0.0_3d'
    descriptor = descriptor[:-11] if "\n_scheduled" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+tv" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+l1" in descriptor else descriptor
    folderPath = trainedNetOffset + "training/denoiser3d/"+[q for q in os.listdir(trainedNetOffset + "training/denoiser3d") if descriptor in q][0]
    fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
    filePath = folderPath + "/" +fileName
    
    if "rdnDenoiser" in descriptor:
        descriptor = descriptor[10:]

    nb_of_features = int([q for q in descriptor.split('_') if 'nF' in q][0][2:])
    nb_of_blocks = int([q for q in descriptor.split('_') if 'nB' in q][0][2:])
    layer_in_each_block = int([q for q in descriptor.split('_') if 'lieb' in q][0][4:])
    growth_rate = int([q for q in descriptor.split('_') if 'gr' in q][0][2:])
    biasFlag = True

    model = rdnDenoiserResRelu3d(input_channels=1,
                nb_of_features=nb_of_features,
                nb_of_blocks=nb_of_blocks,
                layer_in_each_block=layer_in_each_block, 
                growth_rate=growth_rate,
                out_channel=1,
                bias = biasFlag).cuda()

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))
    return model

def getModelForImplicitLD(descriptor, getMaxFlag = True, numIter = 25):
    folderPath = trainedNetOffset + "training/deqmpi/"+descriptor

    fileName = "epoch150END.pth"
    if getMaxFlag:
        fayNeyn = [i for i in os.listdir(folderPath) if "max" in i]
        if len(fayNeyn) > 1:
            jj = -1
            for ffNames in fayNeyn:
                myVal = int(ffNames.split('max')[0][5:])
                if myVal > jj:
                    jj = myVal
            fayNeyn = ["epoch" + str(jj) + "max.pth"]
    else:
        fayNeyn = [i for i in os.listdir(folderPath) if "END" in i]

    if len(fayNeyn) > 0:
        fileName = fayNeyn[0]
        filePath = folderPath + "/" +fileName
    else:
        fayNeyns = os.listdir(folderPath)[0]
        folderPath += "/" + fayNeyns
        fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
        filePath = folderPath + "/" +fileName
    
    splt = descriptor.split("_")

    consistencyDim = int(splt[1][0])

    offset = 2
    nb_of_steps = int(splt[offset + 12])
    nb_of_features = int(splt[offset + 13][2:])
    nb_of_blocks = int(splt[offset + 14][2:])
    layer_in_each_block = int(splt[offset + 15][4:])
    growth_rate = int(splt[offset + 16][2:])
    lambdaVal = float(splt[offset + 17][3:])
    biasFlag = True

    nb_of_featuresL = int(splt[offset + 25])
    nb_of_blocksL = int(splt[offset + 27])
    useNormalizationL = int(splt[offset + 29])

    model2 = rdnLDFixedPt(
        1, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL, bias=True, numDim=2, consistencyDim = consistencyDim).cuda()
    
    model = DEQFixedPoint(model2, fixedPointTekrarlar, max_iter = numIter)

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("num params of model: ", sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))

    return model, lambdaVal

def getModelForADMMLD(descriptor, getMaxFlag = False):
    folderPath = trainedNetOffset + "training/admld/"+descriptor

    fileName = "epoch150END.pth"
    if getMaxFlag:
        fayNeyn = [i for i in os.listdir(folderPath) if "max" in i]
        if len(fayNeyn) > 1:
            jj = -1
            for ffNames in fayNeyn:
                myVal = int(ffNames.split('max')[0][5:])
                if myVal > jj:
                    jj = myVal
            fayNeyn = ["epoch" + str(jj) + "max.pth"]
    else:
        fayNeyn = [i for i in os.listdir(folderPath) if "END" in i]

    if len(fayNeyn) > 0:
        fileName = fayNeyn[0]
        filePath = folderPath + "/" +fileName
    else:
        fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
        filePath = folderPath + "/" +fileName
    
    splt = descriptor.split("_")

    consistencyDim = int(splt[1][0])

    offset = 2
    nb_of_steps = int(splt[offset + 12])
    nb_of_features = int(splt[offset + 13][2:])
    nb_of_blocks = int(splt[offset + 14][2:])
    layer_in_each_block = int(splt[offset + 15][4:])
    growth_rate = int(splt[offset + 16][2:])
    lambdaVal = float(splt[offset + 17][3:])
    biasFlag = True

    nb_of_featuresL = int(splt[offset + 25])
    nb_of_blocksL = int(splt[offset + 27])
    useNormalizationL = int(splt[offset + 29])

    model = rdnADMMLDnet(
        1, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL, bias=True, numDim=2, consistencyDim = consistencyDim).cuda()
    # print(filePath)

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
#     print(sum(pVal.numel() for pVal in model.parameters() if pVal.requires_grad))
    print("num params of model: ", sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))

    return model, lambdaVal


class rdnLDFixedPtScl(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL = 0, bias = True, numDim = 2, consistencyDim = 3, scl = 1):
        super(rdnLDFixedPtScl,self).__init__()
        self.scl = scl
        if numDim == 2:
            self.sharedNet = rdnDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                             layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)
        else:
            self.sharedNet = rdnDenoiserResRelu3d(input_channels=input_channels, nb_of_features=nb_of_features, nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate, out_channel=input_channels, bias = bias)

        if consistencyDim == 0:
            self.consistencyNet = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
        elif consistencyDim < 4:
            self.consistencyNet = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalizationL, numDim = consistencyDim).cuda()
        
    def forward(self, theIn, fixedParams):
        datatC, AtC, MtC, epsilon, x_inShape = fixedParams

        Nbatch = datatC.shape[0]
        Nimg = np.prod(x_inShape[2:])
        x_in, d0, d2 = theIn[:, :Nimg], theIn[:, Nimg:2*Nimg], theIn[:, 2*Nimg:]
        
        Ax = F.linear(x_in, AtC)

        z0 = self.sharedNet((x_in - d0).reshape(x_inShape)).reshape(Nbatch, -1) 

        z2 = self.consistencyNet((Ax - d2).reshape(Nbatch, 1, -1) / self.scl,
                    datatC.reshape(Nbatch, 1, -1) / self.scl, epsilon / self.scl, Ua = 1).reshape(Nbatch, -1) * self.scl

        r = (z0 + d0) + F.linear(z2 + d2, AtC.T)
        x = F.linear(r, MtC)
        Ax = F.linear(x, AtC)
        
        d0 = d0 - x + z0
        d2 = d2 - Ax + z2
        # print("pass time: ",time.time() - tempTime)

        return torch.cat((x.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim = 1 )

#
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return  y + x * self.gamma
class NAFNet(nn.Module):

    def __init__(self, img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class ngcrnanetfixedPt(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, bias=True, numDim=2):
        super(ngcrnanetfixedPt, self).__init__()
        if numDim == 2:  # RDN残差密集块
            self.sharedNet = rdnnafDenoiserResRelu(input_channels=input_channels, nb_of_features=nb_of_features,
                                                nb_of_blocks=nb_of_blocks,
                                                layer_in_each_block=layer_in_each_block, growth_rate=growth_rate,
                                                out_channel=input_channels, bias=bias)


    def forward(self, theIn, fixedParams):
        b, S, lambda_reg, x_inShape = fixedParams  # 固定变量y，A，M，阈值
        # tempTime = time.time()
        Nbatch = b.shape[0]

        Nimg = np.prod(x_inShape[2:])
        M = S.shape[0]
        c, r, z, p = theIn[:, :Nimg], theIn[:, Nimg:Nimg+M], theIn[:, Nimg+M:Nimg+M+Nimg], theIn[:, Nimg+M+Nimg:Nimg+M+2*Nimg]
        for inter in range(3):
            v = torch.matmul(p, S.T)  # 计算搜索方向在系统矩阵下的投影
            alpha = torch.sum(z * z, dim=1) / (torch.sum(v * v, dim=1) + lambda_reg * torch.sum(p * p, dim=1))
            if inter == 1:
                # c = self.sharedNet((c + alpha.unsqueeze(1) * p).reshape(x_inShape)).reshape(Nbatch, -1)

                a = c
                b = alpha.unsqueeze(1) * p
                a = self.sharedNet(a.reshape(x_inShape)).reshape(Nbatch, -1)
                b = self.sharedNet(b.reshape(x_inShape)).reshape(Nbatch, -1)
                c = a + b
            else:
                c = c + alpha.unsqueeze(1) * p  # 更新解

            r = r - alpha.unsqueeze(1) * v  # 更新残差
            z_new = torch.matmul(r, S.conj()) - lambda_reg * c  # 计算新的梯度方向
            beta = torch.sum(z_new * z_new, dim=1) / torch.sum(z * z, dim=1)  # 计算 beta 用于更新搜索方向
            p = z_new + beta.unsqueeze(1) * p  # 更新搜索方向
            # p = self.searchNet(torch.cat((z_new, p, beta.unsqueeze(1)), dim=1))
            z = z_new  # 更新梯度方向
        # c = torch.clamp(c, min=-1e6, max=1e6)

        return torch.cat((c.reshape(Nbatch, -1), r, z, p), dim=1)