import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# ============ 1) 加载数据 ============
X = pd.read_csv("brain_encoding/hrf/CN/bertAlign_regressors.csv", header=None).values  # (n_timepoints, n_features)
n_timepoints = X.shape[0]

bold_img = nib.load('brain_encoding/derivatives-merged/sub-CN001/func/sub-CN001_task-lppCN_mergedTP-2977_bold.nii.gz')
bold_data = bold_img.get_fdata()  # shape: (73, 90, 74, time)
brain_shape = bold_data.shape[:-1]  # (73, 90, 74)
print("Whole brain shape:", brain_shape)

# 取中间一个小立方ROI，比如：
roi_x = (30, 40)
roi_y = (40, 50)
roi_z = (30, 40)
roi_data = bold_data[roi_x[0]:roi_x[1],
                     roi_y[0]:roi_y[1],
                     roi_z[0]:roi_z[1], :]  # shape: (10, 10, 10, time)

# reshape成 (n_voxels, time)
roi_shape = roi_data.shape[:-1]  # (10, 10, 10)
roi_n_voxels = np.prod(roi_shape)
roi_2d = roi_data.reshape(roi_n_voxels, n_timepoints)


def methodA_multi_output(X, roi_2d, n_splits=5, alpha=1.0):
    """ 多输出Ridge，对ROI内所有voxels一起拟合，然后在测试集上计算相关并平均。 """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    brain_shape = roi_2d.shape[0]  # 这里是 n_voxels
    all_r_folds = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = roi_2d[:, train_idx].T  # (train_samples, n_voxels)
        y_test = roi_2d[:, test_idx].T     # (test_samples, n_voxels)

        # 标准化 X，但不动 y
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 多输出 Ridge 回归
        reg = Ridge(alpha=alpha)
        reg.fit(X_train_scaled, y_train)   # y_train shape: (train_samples, n_voxels)

        y_pred = reg.predict(X_test_scaled)   # shape: (test_samples, n_voxels)

        # 逐voxel计算 Pearson r
        # 也可考虑用循环 + pearsonr(y_test[:, v], y_pred[:, v])，但矢量化写法会快一些
        y_test_mean = np.mean(y_test, axis=0, keepdims=True)   # shape: (1, n_voxels)
        y_pred_mean = np.mean(y_pred, axis=0, keepdims=True)

        numerator = np.sum((y_test - y_test_mean) * (y_pred - y_pred_mean), axis=0)
        denominator = np.sqrt( 
            np.sum((y_test - y_test_mean)**2, axis=0) * 
            np.sum((y_pred - y_pred_mean)**2, axis=0)
        ) + 1e-10
        r_fold = numerator / denominator  # shape: (n_voxels,)

        all_r_folds.append(r_fold)
    
    # 对K折做平均
    all_r_folds = np.array(all_r_folds)  # (n_splits, n_voxels)
    mean_r = np.mean(all_r_folds, axis=0)  # shape: (n_voxels,)
    # ROI内所有voxel 再做平均
    roi_mean_r = np.mean(mean_r)
    return roi_mean_r

def methodB_voxel_loop(X, roi_2d, n_splits=5):
    """ 对ROI内每个voxel单独做K折CV，并在所有测试集折拼起来后算一次Pearson r。最后对ROI做平均。 """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_voxels = roi_2d.shape[0]
    voxel_r = []

    for vox_idx in range(n_voxels):
        y = roi_2d[vox_idx, :]  # shape: (timepoints,)

        out_pred = []
        out_true = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 对 X 和 y 都标准化
            X_scaler = StandardScaler()
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1,1)).flatten()

            reg = Ridge(alpha=1.0)
            reg.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = reg.predict(X_test_scaled)

            out_pred.extend(y_pred_scaled)
            out_true.extend(y_test_scaled)

        # 所有折拼接后一次计算相关
        r_all, _ = pearsonr(out_true, out_pred)
        voxel_r.append(r_all)
    
    # ROI内所有 voxel 再做平均
    roi_mean_r = np.mean(voxel_r)
    return roi_mean_r

def compare_two_methods():
    roi_mean_r_A = methodA_multi_output(X, roi_2d, n_splits=5, alpha=1.0)
    roi_mean_r_B = methodB_voxel_loop(X, roi_2d, n_splits=5)

    print("方法A（多输出，一折一算后再平均）ROI平均r:", roi_mean_r_A)
    print("方法B（单voxel拼接后一次性算相关）ROI平均r:", roi_mean_r_B)

compare_two_methods()