import nibabel as nib
from nilearn.image import resample_to_img
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import spatial
import numpy, rsatoolbox

def compute_rdm(data):
    """
    data: numpy array of shape [N, K], 
          N = n_scans (timepoints), 
          K = number of regressors from event2hrf.py (n_features-1).
    We'll compute an N x N RDM using correlation distance: 1 - corr.
    """
    N = data.shape[0]
    rdm = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # 计算第 i 行与第 j 行向量的 Pearson 相关系数
            corr = np.corrcoef(data[i], data[j])[0, 1]
            # 相关距离 = 1 - corr
            rdm[i, j] = 1 - corr
    return rdm


df1 = pd.read_csv("brain_encoding/hrf/EN/llama2_2downsample_regressors.csv", header=None)
data1 = df1.values

df2 = pd.read_csv("brain_encoding/hrf/EN/llama2breakdown_2downsample_regressors.csv", header=None)
data2 = df2.values

d1 = rsatoolbox.data.Dataset(data1)
d2 = rsatoolbox.data.Dataset(data2)
print(data2.shape)
rdms1 = rsatoolbox.rdm.calc_rdm(d1)
rdms2 = rsatoolbox.rdm.calc_rdm(d2)
corr = rsatoolbox.rdm.compare(rdms1,rdms2,method='corr')
print("corr", corr)


# # 2) 计算每行的标准差
# stds1 = np.std(data1, axis=1)

# # 3) 找出标准差为 0 的行（这些行全为常数或全0）
# zero_std_mask1 = (stds1 == 0)
# removed_timepoints = np.where(zero_std_mask1)[0]  # 记录被剔除的行索引
# num_removed1 = len(removed_timepoints)

# # 4) 剔除这些行
# filtered_data1 = data1[~zero_std_mask1]
# N_filtered1 = filtered_data1.shape[0]

# print(f"Found {num_removed1} rows with zero std (out of ).")
# print(f"After removal, we have {N_filtered1} rows.")


# # 2) 计算每行的标准差
# stds2 = np.std(data2, axis=1)

# # 3) 找出标准差为 0 的行（这些行全为常数或全0）
# zero_std_mask2 = (stds2 == 0)
# removed_timepoints = np.where(zero_std_mask2)[0]  # 记录被剔除的行索引
# num_removed2 = len(removed_timepoints)

# # 4) 剔除这些行
# filtered_data2 = data2[~zero_std_mask2]
# N_filtered2 = filtered_data1.shape[0]
# print(filtered_data1.shape)
# print(N_filtered2)

# cosins = np.zeros(N_filtered2)
# pearsons = np.zeros(N_filtered2)

# for i in range(N_filtered2):
#     data1_flat = filtered_data1[i].flatten()
#     data2_flat = filtered_data2[i].flatten()
#     cos_sim = 1 - spatial.distance.cosine(data1_flat, data2_flat)
#     cosins[i] = cos_sim
#     pearson_corr = np.corrcoef(data1_flat, data2_flat)[0, 1]
#     pearsons[i] = pearson_corr



# print("cosine",np.mean(cosins))

# print("pearson_corr",np.mean(pearsons))

    


# print(data2.shape)

# data1_flat = data1.flatten()
# data2_flat = data2.flatten()

# print(data2_flat.shape)

# # cos_sim = 1 - spatial.distance.cosine(data1_flat, data2_flat)
# # print("cosine",cos_sim)
# cos_sim = data1_flat.dot(data2_flat) / (np.linalg.norm(data1_flat) * np.linalg.norm(data2_flat))
# print(cos_sim)

# pearson_corr = np.corrcoef(data1_flat, data2_flat)[0, 1]
# print("pearson_corr", pearson_corr)







# atlas_img = nib.load("brain_encoding/atlas/SPM/LanA_n806.nii")  # 原始语言图
# bold_img  = nib.load("brain_encoding/ds003643-download/derivatives/sub-CN001/func/sub-CN001_task-lppCN_run-04_space-MNIColin27_desc-preproc_bold.nii.gz")

# atlas_resampled_img = resample_to_img(
#     source_img=atlas_img,
#     target_img=bold_img,
#     interpolation='linear'
# #    interpolation='nearest'
# )

# atlas_resampled_img.to_filename("brain_encoding/atlas/SPM/LanA_n806_inMNIColin27.nii")


# atlas_path = "brain_encoding/atlas/SPM/LanA_n806.nii"
# atlas_img = nib.load(atlas_path)

# # 获取图像数据的 numpy array
# atlas_data = atlas_img.get_fdata()

# print("Atlas shape:", atlas_data.shape)    # e.g., (91, 109, 91)
# print("Atlas affine:\n", atlas_img.affine)
# print("Atlas header:\n", atlas_img.header)


# bold_path = "brain_encoding/ds003643-download/derivatives/sub-CN001/func/sub-CN001_task-lppCN_run-04_space-MNIColin27_desc-preproc_bold.nii.gz"
# bold_img = nib.load(bold_path)
# bold_data = bold_img.get_fdata()

# print("BOLD shape:", bold_data.shape)  # e.g., (73, 90, 74, n_timepoints)
# print("BOLD affine:\n", bold_img.affine)
# print("BOLD header:\n", bold_img.header)