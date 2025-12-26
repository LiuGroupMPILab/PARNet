import numpy as np
from scipy.sparse.linalg import gmres
import h5py
from PIL import Image
from scipy.io import loadmat
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from skimage.transform import iradon
import torch
from skimage.transform import radon

def image_to_sinogram(image, num_angles=180):
    # 生成投影角度
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    # Radon 变换
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram

def load_mat_file(mat_path: str, psf_key: str = 'u_triple_f_D') -> np.ndarray:
    """
    从 MATLAB .mat 文件中加载数据
    - v7.3 及以上：使用 h5py
    - v7 及以下：使用 scipy.io.loadmat
    """
    try:
        # 尝试用 scipy 加载（适合 v7 及以下版本）
        data = loadmat(mat_path)
        print(data.keys())
        if psf_key in data:
            psf = data[psf_key]
            psf = psf.squeeze()
            return psf
        else:
            raise KeyError(f"Key '{psf_key}' not found in {mat_path} using loadmat.")
    except NotImplementedError:
        # v7.3 及以上版本，用 h5py
        with h5py.File(mat_path, 'r') as f:
            if psf_key not in f:
                raise KeyError(f"Key '{psf_key}' not found in {mat_path} using h5py.")
            psf = np.array(f[psf_key])
            psf = psf.squeeze()
            return psf
def deconv_gmres(image_path: str, mat_path: str, tol=8e-2, maxiter=None) -> np.ndarray:
    psf = load_mat_file(mat_path,'u_triple_f_D')
    image = load_mat_file(image_path, 'sinogram_abs_3f_D')
    N_shift, N_rotation = image.shape
    SM = np.zeros((N_shift, 2*N_shift), dtype=np.float32)

    # 构建卷积矩阵
    # for i in range(N_shift):
    #     SM[i, i:i+N_shift] = psf

    # 截取中心部分，生成 N_shift x N_shift 的系统矩阵
    # start_idx = N_shift // 2
    # SM = SM[:, start_idx:start_idx+N_shift]
    #
    # # 初始化输出
    # deconv_image = np.zeros_like(image, dtype=np.float32)
    #
    # # 每列求解
    # for i in range(N_rotation):
    #     x, exitCode = gmres(SM, image[:, i], tol=tol, restart=maxiter)
    #     deconv_image[:, i] = x

    return image




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

#对图片拉东变换再进行FBP，研究稀疏角重建
def dconvfbp(input_tensor):
    """
    input_tensor: torch.Tensor, shape [B, C, H, W]
    output:
        fbpimage: Tensor [B, 1, 100, 100]
        sinogramlist: Tensor [B, 1, H', W']
    """

    # ----------- 安全检查 -----------
    assert isinstance(input_tensor, torch.Tensor), "Input must be a tensor"
    assert input_tensor.dim() == 4, "Input shape must be [B, C, H, W]"
    B, C, H, W = input_tensor.shape
    assert C == 1, "This function currently only supports 1-channel images"

    # ----------- 转 numpy，保持 batch 循环 -----------
    imgs_np = input_tensor.squeeze(1).detach().cpu().numpy().astype(np.uint8)

    resize101 = transforms.Compose([
        transforms.Resize([101, 101], antialias=True),
        transforms.ToTensor()
    ])

    # 加载 PSF
    psf = load_mat_file('./datasets/ground/PSF_101shift_80angles_nb_70.mat', 'u_triple_f_D')

    imagelist = []
    sinogramlist = []

    # ============================
    #        逐张处理
    # ============================
    for b in range(B):
        img = imgs_np[b]
        img_pil = Image.fromarray(img)

        img_resized = resize101(img_pil).squeeze(0).numpy()

        # ----------- 生成 sinogram -----------
        sinogram = image_to_sinogram(img_resized, 180)

        image = sinogram
        N_shift, N_rotation = image.shape
        SM = np.zeros((N_shift, 2 * N_shift), dtype=np.float32)

        # ----------- 构建系统矩阵 -----------
        for i in range(N_shift):
            SM[i, i:i + N_shift] = psf

        start_idx = N_shift // 2
        SM = SM[:, start_idx:start_idx + N_shift]

        # ----------- GMRES 求解 -----------
        deconv_image = np.zeros_like(image, dtype=np.float32)
        for i in range(N_rotation):
            x, exitCode = gmres(SM, image[:, i], tol=8e-2, restart=None)
            deconv_image[:, i] = x

        image_degraded = image.copy()
        sinogramlist.append(image_degraded)

        # ----------- FBP 重建 -----------
        FBPimage = FBP(psf, image_degraded, False)
        imagelist.append(FBPimage)

    # ----------- 输出 Resize 成 100x100 -----------
    resize100 = transforms.Resize([100, 100], antialias=True)

    imagelist = np.array(imagelist)[:, None, :, :]  # [B, 1, H, W]
    fbp_tensor = torch.from_numpy(imagelist).float()
    fbp_tensor = resize100(fbp_tensor)

    sinogramlist = np.array(sinogramlist)[:, None, :, :]
    sinogram_tensor = torch.from_numpy(sinogramlist).float()

    return fbp_tensor.to("cuda")


def dconvfbp_tensor(input_tensor, psf_tensor=None):
    """
    input_tensor: torch.Tensor, shape [B, 1, H, W], float32, on CPU or GPU
    psf_tensor: torch.Tensor, shape [N_shift], optional, float32, on same device as input
    output:
        fbp_tensor: Tensor [B, 1, 100, 100], same device as input
        sinogram_tensor: Tensor [B, 1, H', W'], same device as input
    """
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.dim() == 4 and input_tensor.size(1) == 1

    device = input_tensor.device
    B, C, H, W = input_tensor.shape

    # ----------- Resize 到 101x101 -----------
    img_resized = F.interpolate(input_tensor, size=(101, 101), mode='bilinear', align_corners=False)  # [B,1,101,101]

    # ----------- 生成 sinogram (逐张或批量) -----------
    sinogram_list = []
    deconv_list = []
    fbp_list = []

    if psf_tensor is None:
        psf_np = load_mat_file('./datasets/ground/PSF_101shift_80angles_nb_70.mat', 'u_triple_f_D')
        psf_tensor = torch.tensor(psf_np, dtype=torch.float32, device=device)
    N_shift = psf_tensor.numel()
    N_rotation = 180  # 固定角度

    # 构建系统矩阵 SM，一次性
    SM = torch.zeros(N_shift, 2 * N_shift, dtype=torch.float32, device=device)
    for i in range(N_shift):
        SM[i, i:i+N_shift] = psf_tensor
    start_idx = N_shift // 2
    SM = SM[:, start_idx:start_idx+N_shift]  # [N_shift, N_shift]

    # 逐张处理
    for b in range(B):
        img = img_resized[b,0]  # [H,W]
        sinogram = image_to_sinogram(img.cpu().numpy(), N_rotation)  # 还是用原函数生成 sinogram
        sinogram_tensor_b = torch.tensor(sinogram, dtype=torch.float32, device=device)  # [N_shift, N_rotation]
        sinogram_list.append(sinogram_tensor_b)

        # ----------- GMRES 求解 (CPU) 或换成 lstsq GPU ------------
        deconv = torch.zeros_like(sinogram_tensor_b)
        for i in range(N_rotation):
            x, exitCode = gmres(SM.cpu().numpy(), sinogram[:,i], tol=8e-2, restart=None)
            deconv[:,i] = torch.tensor(x, dtype=torch.float32, device=device)
        deconv_list.append(deconv)

        # ----------- FBP 重建 (CPU) -----------
        FBP_img = FBP(psf_tensor.cpu().numpy(), deconv.cpu().numpy(), False)
        fbp_list.append(torch.tensor(FBP_img, dtype=torch.float32, device=device))

    # ----------- 转成 tensor -----------
    fbp_tensor = torch.stack(fbp_list, dim=0)[:,None,:,:]  # [B,1,H,W]
    fbp_tensor = F.interpolate(fbp_tensor, size=(100,100), mode='bilinear', align_corners=False)
    sinogram_tensor = torch.stack(sinogram_list, dim=0)[:,None,:,:]  # [B,1,N_shift,N_rotation]

    return fbp_tensor.to(device)