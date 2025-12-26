import numpy as np
from skimage.transform import iradon
from scipy.sparse.linalg import gmres
from utils.FBP_utils import load_mat_file
import os
def FBP(PSF, sinogram, flag_deconv=False):
    N_shift, N_rotation = sinogram.shape
    if flag_deconv:
        # 生成FBP的系统矩阵
        SM_FBP = np.zeros((N_shift, 2 * N_shift))  # 创建全零矩阵
        for i_shift in range(N_shift):
            SM_FBP[i_shift, i_shift:i_shift + N_shift] = PSF  # 将 PSF 填充到对角线区域
        SM_FBP = SM_FBP[:, N_shift // 2: N_shift // 2 + N_shift]

        # 去卷积的sinogram
        sinogram_deconv = np.zeros_like(sinogram)
        for i_rotation in range(N_rotation):
            sinogram_deconv[:, i_rotation], info = gmres(SM_FBP, sinogram[:, i_rotation], tol=8e-2, maxiter=N_shift)
            # sinogram_deconv[sinogram_deconv < 0] = 0.0 * sinogram_deconv[sinogram_deconv < 0]
            # sinogram_deconv[:, i_rotation],_ = nnls(SM_FBP, sinogram[:, i_rotation], maxiter=N_shift)
        image = iradon(sinogram_deconv, filter_name='hamming', interpolation='linear', output_size=N_shift, circle=False)  # 滤波器：Ram-Lak（低噪声）、Shepp-Logan（中噪声）、Hann（高噪声）、Hamming（更高噪声）
    else:
        image = iradon(sinogram, filter_name='hamming', interpolation='linear', output_size=N_shift, circle=False)
    return image


def fbp_singel(path1=r'.\datasets\ground\PSF_101shift_80angles_nb_70.mat', path2=r'./datasets/ground/sinogram_101shift_80angles_U.mat'):
    psf_path = path1
    sinogram_path = path2
    psf = load_mat_file(psf_path,'u_triple_f_D')
    sinogram = load_mat_file(sinogram_path, 'proj_mag_crop_bc') # IR_crop proj_mag_crop_bc sinogram_abs_3f_D

    rec_img = FBP(psf, sinogram, flag_deconv=False)

    return rec_img


def Sinogram_mean_fbp(filepath='datasets/ground/P'):
    sinogram_list = []
    for file_name in os.listdir(filepath):
        file_path = os.path.join(filepath, file_name)
        if file_path.endswith('.mat'):
            # 假设 mat 文件中 sinogram 的变量名是 'sinogram'
            sinogram = load_mat_file(file_path, 'proj_mag_crop_bc')
            sinogram_list.append(sinogram)
    sinogram_mean = np.mean(sinogram_list, axis=0)

    psf = load_mat_file('.\datasets\ground\PSF_101shift_80angles_nb_70.mat', 'u_triple_f_D')
    rec_img = FBP(psf, sinogram_mean, flag_deconv=False)



    return rec_img
