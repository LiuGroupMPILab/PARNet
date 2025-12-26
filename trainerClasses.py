# from cv2 import UMAT_SUBMATRIX_FLAG
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from torch import nn
import torch.nn.functional as F
import time
from modelClasses import computeRowEnergy
from torch.utils.data import DataLoader
import wandb
import torch
from torchvision import transforms
from utils.utils import ssim_index, compute_ssim_batch
from utils.mean_SD import calculate_image_metrics,calculate_image_metricsv2
from utils.FBP_utils import dconvfbp
import matplotlib.pyplot as plt
from PIL import Image

import logging


def setup_logger(model_name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{model_name}_train_log.txt")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='w'
    )
    logging.info(f"Logger initialized for model: {model_name}")


def transformDataset(data, imgSizes, rescaleVals, randVals=None):  # imgSizes(26,13)

    dims = len(imgSizes)  # 2

    reScaleMin, reScaleMax = rescaleVals  #
    imgSize = data.shape  #

    n1, n2 = imgSizes
    data -= data.reshape(imgSize[0], imgSize[1], -1).min(dim=2).values[:, :, None, None]
    data /= data.reshape(imgSize[0], imgSize[1], -1).max(dim=2).values[:, :, None, None]

    if (n1 < data.shape[2]) or (n2 < data.shape[3]):
        if randVals is None:
            rand1 = torch.randint(low=0, high=data.shape[2] - n1, size=(1,))
            rand2 = torch.randint(low=0, high=data.shape[3] - n2, size=(1,))
        else:
            rand1 = randVals[0]
            rand2 = randVals[1]
    else:
        rand1 = 0
        rand2 = 0
    data = data[:, :, rand1:rand1 + n1, rand2:rand2 + n2]

    if reScaleMax > 0:
        randScale = torch.rand((imgSize[0], 1, 1, 1), device=data.device) * reScaleMax + reScaleMin
        if dims == 3:
            randScale = randScale.reshape(imgSize[0], 1, 1, 1, 1)
    else:
        randScale = 1

    data *= randScale
    return data



transform = transforms.Compose([
    transforms.Resize([100, 100], antialias=True),  # 调整图片大小
])
def trainDenoiserPAR(model, epoch_nb, loss, optimizer, scheduler, trainDataset, valDataset, window_size, batch_size_train,
                        batch_size_val, rescaleVals=[1, 1], saveModelEpoch=0, valEpoch=0, saveDirectory='',
                        optionalMessage="", wandbFlag=False, dims=2):
    trainLosses = torch.zeros(epoch_nb)
    trainNrmses = torch.zeros(epoch_nb)
    trainPsnrs = torch.zeros(epoch_nb)
    trainSSIMs = torch.zeros(epoch_nb)
    valLosses = list()
    valNrmses = list()
    valPsnrs = list()
    valSSIMs = list()
    trainLoader = DataLoader(trainDataset, batch_size_train, shuffle=False)
    valLoader = DataLoader(valDataset, valDataset.__len__(), shuffle=False)
    reScaleMin, reScaleMax = rescaleVals
    setup_logger("PAR")
    for epoch in range(1, 1 + int(epoch_nb)):
        tempLosses = list()
        model.train()
        tempNrmseNumeratorSquare = 0
        tempNrmseDenumeratorSquare = 0
        tempNumel = 0
        tempTime = time.time()

        if saveModelEpoch > 0:
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), saveDirectory + r"/" + optionalMessage + "epoch" + str(epoch) + ".pth")
        idx = -1
        for noisyInp, data in trainLoader:
            idx += 1
            data = transform(data)  # * mask
            data = data.float().cuda()
            noisyInp = transform(noisyInp)  # * mask
            noisyInp = noisyInp.float().cuda()



            data = transformDataset(data, [*data.shape[2:]], rescaleVals)
            noisyInp = transformDataset(noisyInp, [*noisyInp.shape[2:]], rescaleVals)

            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = noisyInp.size()
            window_size = window_size
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            noisyInp = torch.cat([noisyInp, torch.flip(noisyInp, [2])], 2)[:, :, :h_old + h_pad, :]
            noisyInp = torch.cat([noisyInp, torch.flip(noisyInp, [3])], 3)[:, :, :, :w_old + w_pad]

            modelOut = model(noisyInp)
            modelOut = modelOut[..., :h_old * 1, :w_old * 1]

            model.zero_grad()
            model_loss = loss(modelOut, data)
            # model_loss.backward()
            optimizer.step()
            if dims == 3:
                percent = 10
                if idx % int(trainDataset.__len__() / batch_size_train / percent) == 0:
                    print('Epoch {0:d} | {1:d}% | batch nrmse: {2:.5f}'.format(epoch, percent * idx // int(
                        trainDataset.__len__() / batch_size_train / percent), (float(torch.norm(modelOut - data))) / (
                                                                                   float(torch.norm(data)))))  # myflag

            with torch.no_grad():
                tempLosses.append(float(model_loss))
                tempNrmseNumeratorSquare += (float(torch.norm(modelOut - data))) ** 2
                tempNrmseDenumeratorSquare += (float(torch.norm(data))) ** 2
                tempNumel += modelOut.numel()

        #     print(f"Load: Opt: {t4 - t0:.3f}s")
        # print(f"Epoch time: {time.time() - start:.2f}s")
        # back to epoch
        model.eval()
        scheduler.step()

        trainLosses[epoch - 1] = sum(tempLosses) / len(tempLosses)
        trainNrmses[epoch - 1] = (tempNrmseNumeratorSquare / tempNrmseDenumeratorSquare) ** (1 / 2)
        trainPsnrs[epoch - 1] = 20 * \
                                torch.log10(1 / (tempNrmseDenumeratorSquare ** (
                                        1 / 2) *  # Should we correct 1 -> valGround.max()
                                                 trainNrmses[epoch - 1] / (tempNumel) ** (1 / 2)))
        trainSSIMs[epoch - 1] = float(ssim_index(modelOut, data, data_range=1.0))
        epochTime = time.time() - tempTime
        if wandbFlag:
            wandb.log({"train_loss": trainLosses[epoch - 1], "train_nrmse": trainNrmses[epoch - 1],
                       "train_psnr": trainPsnrs[epoch - 1]})
        logging.info(
            f"Epoch: {epoch}, Train Loss = {trainLosses[epoch - 1]:.6f}, Train nRMSE = {trainNrmses[epoch - 1]:.6f},\
                      Train pSNR = {trainPsnrs[epoch - 1]:.6f}, time elapsed = {epochTime:.6f}")
        print(
            "Epoch: {0}, Train Loss = {1:.6f}, Train nRMSE = {2:.6f}, Train pSNR = {3:.6f}, Train SSIM = {4:.6f}, time elapsed = {5:.6f}".format(
                epoch,
                trainLosses[epoch - 1], trainNrmses[epoch - 1], trainPsnrs[epoch - 1], trainSSIMs[epoch - 1],
                epochTime))

        if valEpoch > 0:
            if epoch % valEpoch == 0:
                with torch.no_grad():
                    model.eval()
                    valInpVal, valInp = next(iter(valLoader))  # * mask
                    valInp = transform(valInp)
                    valGround = valInp.clone()
                    valInpVal = transform(valInpVal)

                    valGround = transformDataset(valGround, [*valInp.shape[2:]], rescaleVals)
                    valInpVal = transformDataset(valInpVal, [*valInp.shape[2:]], rescaleVals)
                    _, _, h_old, w_old = valInpVal.size()
                    window_size = window_size
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    valInpVal = torch.cat([valInpVal, torch.flip(valInpVal, [2])], 2)[:, :, :h_old + h_pad, :]
                    valInpVal = torch.cat([valInpVal, torch.flip(valInpVal, [3])], 3)[:, :, :, :w_old + w_pad]

                    valOut = torch.zeros_like(valGround)
                    deviceVal = valGround.device  # 'cuda' if valOut.is_cuda else 'cpu'
                    iii = 0
                    while (iii < valInpVal.shape[0] - (valInpVal.shape[0] % batch_size_val)):
                        valInpC = valInpVal[iii:iii + batch_size_val].float().cuda()
                        valOut[iii:iii + batch_size_val] = model(valInpC)[..., :h_old * 1, :w_old * 1].to(deviceVal)
                        iii += batch_size_val

                    valInpC = valInpVal[iii:].float().cuda()


                    valLoss = float(nn.L1Loss()(valGround, valOut))
                    valNrmse = float(torch.norm(valGround - valOut) / torch.norm(valGround))
                    valPSNR = float(20 * \
                                    torch.log10(1 / (torch.norm(valGround) *  # Should we correct 1 -> valGround.max()
                                                     valNrmse / (valOut.numel()) ** (1 / 2))))
                    valSSIM = float(ssim_index(valOut, valGround, data_range=1.0))
                    # valPSNR_avg = (20 *
                    #                 torch.log10(1 / (torch.norm(valGround-valOut, dim = (2, 3)).squeeze() / (valOut[0,0].numel()) ** (1/2))))
                    valPSNR_avg = (20 *
                                   torch.log10(1 / (torch.norm(
                                       valGround.reshape(valGround.shape[0], -1) - valOut.reshape(valGround.shape[0],
                                                                                                  -1),
                                       dim=(1)).squeeze() / (valOut[0, 0].numel()) ** (1 / 2))))

                    valLosses.append(valLoss)
                    valNrmses.append(valNrmse)
                    valPsnrs.append(valPSNR)
                if wandbFlag:
                    wandb.log({"valid_nRMSE": valNrmse,
                               "ref_nRMSE": torch.norm(valInp - valGround) / torch.norm(valGround),
                               'valid_pSNR': valPSNR,
                               'valid_pSNRavg': valPSNR_avg.mean(0),
                               'valid_pSNRstd': valPSNR_avg.std(0),
                               'valid_loss': valLoss})
                logging.info(f"Epoch: {epoch}, Val Loss = {valLoss:.6f}, Val nRMSE = {valNrmse:.6f},\
                      Val pSNR = {valPSNR:.6f}")
                print(
                    "---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}, Val SSIM {4:.6f}".format(
                        epoch, valLoss,
                        valNrmse,
                        valPSNR,
                        valSSIM))

    if wandbFlag:
        wandb.log({"valid_nRMSE": valNrmse,
                   'valid_pSNR': valPSNR,
                   'valid_pSNRavg': valPSNR_avg.mean(0),
                   'valid_pSNRstd': valPSNR_avg.std(0),
                   'valid_loss': valLoss})
    print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}, Val SSIM {4:.6f}".format(epoch,
                                                                                                                valLoss,
                                                                                                                valNrmse,
                                                                                                                valPSNR,
                                                                                                                valSSIM))
    torch.save(model.state_dict(), saveDirectory + r"/" + optionalMessage + "epoch" + str(epoch) + "END.pth")
    return model, [trainLosses.numpy(), trainNrmses.numpy(), trainPsnrs.numpy()], [np.array(valLosses),
                                                                                   np.array(valNrmses),
                                                                                   np.array(valPsnrs)]


