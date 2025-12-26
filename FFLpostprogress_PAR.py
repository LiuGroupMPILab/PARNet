import wandb
from torch.utils.data import Subset
from data import *
from trainerClasses import *
from models.PARV1 import PAR
import os
from fvcore.nn import FlopCountAnalysis
from utils.measure_single_image_latency import measure_single_image_latency
import time
import torch
from torch import nn
import numpy as np
import argparse
import sys
# from utils.utils import CustomLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时规避 OpenMP 冲突
parser = argparse.ArgumentParser(description="Rdn grid search")

parser.add_argument("--useGPU", type=int, default=0,
                    help="GPU ID to be utilized")

parser.add_argument("--wd", type=float, default=0,
                    help='weight decay')

parser.add_argument("--lr", type=float,
                    default=1e-3, help='learning rate')

parser.add_argument("--saveModelEpoch", type=int,
                    default=1, help="Save model per epoch")

parser.add_argument("--valEpoch", type=int, default=1,
                    help="compute validation per epoch")

parser.add_argument("--fixedNsStdFlag", type=int, default=1,
                    help='0: randomly generate noise std for each image, 1: fix noise std.')

parser.add_argument("--minNoiseStd", type=float, default=0, help='For non-fixed noise, minimum noise std.')

parser.add_argument("--maxNoiseStdList", type=str, default='0.05',
                    help='For non-fixed noise: maximum noise std., For fixed noise: noise std.')

parser.add_argument("--batch_size_train", type=int, default=4,
                    help="Batch Size")
parser.add_argument("--window_size", type=int, default=8,
                    help="SwinIR window_size")
parser.add_argument("--epoch_nb", type=int, default=405,
                    help="When to decay learning rate; should be less than epochs")

parser.add_argument("--wandbFlag", type=int, default=0, help="use wandb = 1 for tracking loss")
parser.add_argument("--wandbName", type=str,
                    default="ppmpi", help='experiment name for WANDB')

parser.add_argument("--reScaleBetween", type=str, default="1,1",
                    help='scale images randomly between')

parser.add_argument("--dims", type=int, default=2, help='Number of dimensions of the denoiser')

parser.add_argument("--nb_of_featuresList", type=str,
                    default="12",
                    help='Number of features of RDN, separate with comma for training of multiple different networks')
parser.add_argument("--nb_of_blocks", type=int,
                    default=4, help='Number of blocks of RDN')
parser.add_argument("--layer_in_each_block", type=int,
                    default=4, help='Layer in each block of RDN')
parser.add_argument("--growth_rate", type=int, default=12,
                    help='growth rate of RDN')

opt = parser.parse_args()
print(opt)
Imagesize = [32, 32]
asel = True
dims = opt.dims
resultFolder = "training/denoiser" if dims == 2 else "training/denoiser3d"

useGPUno = opt.useGPU
torch.cuda.set_device(useGPUno)

batch_size_train = opt.batch_size_train
weight_decay = opt.wd
lr = opt.lr
layer_in_each_block = opt.layer_in_each_block
nb_of_blocks = opt.nb_of_blocks
growth_rate = opt.growth_rate
nb_of_featuresList = np.array(opt.nb_of_featuresList.split(',')).astype(int)
window_size = opt.window_size
batch_size_val = batch_size_train
epoch_nb = opt.epoch_nb
saveModelEpoch = opt.saveModelEpoch
wandbFlag = bool(opt.wandbFlag)
minNoiseStd = opt.minNoiseStd
maxNoiseStdList = np.array(opt.maxNoiseStdList.split(',')).astype(float)
valEpoch = opt.valEpoch
wandbProjectName = opt.wandbName
fixedNsStdFlag = bool(opt.fixedNsStdFlag)

mraFolderPath = "datasets/"

reScaleBetween = np.array(opt.reScaleBetween.split(",")).astype(float)

reScaleMin = reScaleBetween[0]
reScaleMax = reScaleBetween[1] - reScaleBetween[0]

tmpTm3 = time.time()
trainDataset = MRAdatasetH5NoScale(mraFolderPath + 'train_FBPFigure.h5',mraFolderPath + 'train_PreFBPFigure.h5',
                                      prefetch=True, dim=2, device=torch.device('cuda'))
indices = range(4)  # 选择前 1000 个样本
trainDataset = Subset(trainDataset, indices)
print('It takes {0:.2f} seconds from train set to RAM'.format(time.time() - tmpTm3))  # myflag
print('Train set size:', trainDataset.__len__())
tmpTm4 = time.time()

valDataset = MRAdatasetH5NoScale(mraFolderPath + 'val_FBPFigure.h5',mraFolderPath + 'val_PreFBPFigure.h5',
                                      prefetch=True, dim=2, device=torch.device('cuda'))
indicesval = range(1000)  # 选择前 1000 个样本
valDataset = Subset(valDataset, indicesval)
print('It takes {0:.2f} seconds from val set to GPU'.format(time.time() - tmpTm4))  # myflag
print('Validation set size:', valDataset.__len__())

for nb_of_features in nb_of_featuresList:
    for maxNoiseStd in maxNoiseStdList:
        tempStr = "PARFigure_lr_" + str(lr) + "_wd_" + str(weight_decay) + "_bs_" \
                  + str(batch_size_train) + \
                  '_gr' + str(growth_rate) + \
                  "depths=[6, 6, 6, 6, 6, 6], embed_dim=30 ,num_heads=[6, 6, 6, 6, 6, 6],"

        tempStr = tempStr if dims == 2 else tempStr + "_3d"

        saveFolder = resultFolder + "/" + tempStr
        optionalMessage = ""

        if wandbFlag:
            wandb.init(project=wandbProjectName, reinit=True, name=tempStr)

        # print(opt)
        # print(optionalMessage)

        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        model = PAR(upscale=1, in_chans=1, img_size=104, window_size=window_size,
                                                img_range=255., depths=[6, 6,6,6,6,6], embed_dim=30,
                                                num_heads=[6, 6,6,6,6,6],
                                                mlp_ratio=2, upsampler='', resi_connection='1conv').cuda()

        preLoadDir = "training/Alphanumeric/epoch300END.pth"
        model.load_state_dict(torch.load(preLoadDir, map_location=next(model.parameters()).device))
        print("number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # loss = CustomLoss(alpha=0.5).to("cuda")
        loss = nn.L1Loss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_nb // 5, gamma=0.5)

        model, trainMetrics, valMetrics = trainDenoiserPAR(model=model,
                                                              epoch_nb=epoch_nb,
                                                              loss=loss,
                                                              optimizer=optimizer,
                                                              scheduler=scheduler,
                                                              trainDataset=trainDataset,
                                                              valDataset=valDataset,
                                                              window_size=window_size,
                                                              batch_size_train=batch_size_train,
                                                              batch_size_val=batch_size_val,
                                                              rescaleVals=[reScaleMin, reScaleMax],
                                                              saveModelEpoch=saveModelEpoch,
                                                              valEpoch=valEpoch,
                                                              saveDirectory=saveFolder,
                                                              optionalMessage=optionalMessage,
                                                              wandbFlag=wandbFlag,
                                                              dims=dims)