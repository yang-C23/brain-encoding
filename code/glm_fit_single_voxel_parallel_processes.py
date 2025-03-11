import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
import pickle
from scipy.stats import pearsonr
from joblib import Parallel, delayed  # 引入 joblib

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_model_Ridge(X, y, n, saveto, save_results = True, shuffle=False, prefix = ""):
    if shuffle: 
        kf = KFold(n_splits=n, shuffle=True, random_state=0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    out_coefs = []
    out_pred = []
    y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in kf.split(X):
        y_train_orig = y[train_index]
        y_test_orig = y[test_index]
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=0.1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test.tolist())
        coefs = reg.coef_
        out_coefs.append(coefs)
        out_reg.append(r)
    r_tot = pearsonr(out_pred, y_tot)[0]
    print(round(r_tot, 4))
    out_predictions = [y_tot, out_pred]
    if save_results:
        save(r_tot, f"results/rs/{prefix}{saveto}")
        save(out_reg, f"results/out_reg/{prefix}{saveto}")
        save(out_coefs, f"results/coefficients/{prefix}{saveto}")
        save(out_predictions, f"results/predictions/{prefix}{saveto}")
    return r_tot

# 读取数据
X = pd.read_csv("brain_encoding/hrf/CN/bertAlign_regressors.csv", header=None).values
n_timepoints = X.shape[0]

bold_img = nib.load('brain_encoding/derivatives-merged/sub-CN001/func/sub-CN001_task-lppCN_mergedTP-2977_bold.nii.gz')
bold_data = bold_img.get_fdata()
brain_shape = bold_data.shape[:-1]  # (73, 90, 74)
print(brain_shape)

n_voxels = np.prod(brain_shape)
bold_2d = bold_data.reshape(n_voxels, n_timepoints)

# 过滤掉无效voxel
# voxel_stds = np.std(bold_2d, axis=1)
# threshold = 1e-5
# mask = voxel_stds > threshold
# valid_voxel_indices = np.where(mask)[0]
# print("有效 voxel 数量:", len(valid_voxel_indices), "/", n_voxels)

r_map = np.full(brain_shape, np.nan, dtype=np.float32)

# 定义一个处理单个 voxel 的函数
def process_voxel(vox_idx):
    y = bold_2d[vox_idx, :]
    # 此处设置 save_results=False，避免在并行计算时写文件冲突
    r_tot = test_model_Ridge(X, y, 5, saveto=f"voxel_{vox_idx}", save_results=False, shuffle=False)
    return vox_idx, r_tot

# 使用 joblib 并行化处理所有有效的 voxel
results = Parallel(n_jobs=-1)(
    delayed(process_voxel)(vox_idx) for vox_idx in tqdm(n_voxels, desc="Processing voxels")
)

# 将结果写入 r_map
for vox_idx, r_tot in results:
    r_map.flat[vox_idx] = r_tot

brain_mean_r = np.nanmean(r_map)
print("Whole brain average r:", brain_mean_r)
