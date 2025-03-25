import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ========== 新增：pearsonr ==========
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description='语言区编码分析')
    parser.add_argument('--hrf_csv', type=str, required=True,
                      help='HRF模型文件路径')
    parser.add_argument('--derivatives_dir', type=str, required=True,
                      help='BOLD数据根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='结果输出目录')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='Ridge回归的正则化系数')
    parser.add_argument('--lang_atlas', type=str, required=True,
                      help='语言概率图 (SPM_LanA_n806.nii) 的路径')
    parser.add_argument('--threshold', type=float, default=0.2,
                      help='语言图阈值 (默认0.2)')
    parser.add_argument('--output_nii', action='store_true',
                    help='如指定此参数，则输出nii.gz结果文件')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 从路径提取语言代码
    lang = os.path.normpath(args.hrf_csv).split(os.sep)[-2]
    hrf_model = os.path.basename(args.hrf_csv).split('_')[0]

    # 查找匹配的被试
    subject_dirs = glob.glob(os.path.join(args.derivatives_dir, f'sub-{lang}*'))
    subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
    
    if not subjects:
        raise ValueError(f"未找到{lang}语言的被试数据")

    # 加载设计矩阵 (X)
    X = pd.read_csv(args.hrf_csv, header=None).values  # shape: (n_timepoints, n_features)
    n_timepoints = X.shape[0]

    # =========== 加载语言概率图，并创建二值掩码 ===========
    atlas_img = nib.load(args.lang_atlas)
    atlas_data = atlas_img.get_fdata()  # shape: (X, Y, Z)
    language_mask = (atlas_data >= args.threshold)      # boolean mask

    # 确保语言图只有 3 维，与 BOLD 的空间维度对应
    if len(atlas_data.shape) != 3:
        raise ValueError("语言图 (SPM_LanA_n806.nii) 应该是3D的，请检查文件。")

    results = []

    # 交叉验证配置
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    for sub_id in tqdm(subjects, desc='处理被试'):
        bold_pattern = os.path.join(
            args.derivatives_dir, sub_id, 'func',
            f'{sub_id}_task-lpp*_mergedTP-*_bold.nii.gz'
        )
        bold_files = glob.glob(bold_pattern)
        
        if not bold_files:
            print(f"\n跳过 {sub_id}: 未找到BOLD文件")
            continue
            
        try:
            bold_img = nib.load(bold_files[0])
            bold_data = bold_img.get_fdata()  # (X, Y, Z, n_timepoints)
        except Exception as e:
            print(f"\n{sub_id}加载BOLD数据失败: {str(e)}")
            continue

        # 验证时间维度
        if bold_data.shape[-1] != n_timepoints:
            print(f"\n跳过 {sub_id}: 时间点不匹配")
            continue
        
        # 验证空间维度是否与atlas一致
        if bold_data.shape[:3] != atlas_data.shape:
            print(f"\n跳过 {sub_id}: BOLD与语言图的空间维度不匹配")
            continue

        brain_shape = bold_data.shape[:-1]  # e.g. (73, 90, 74)
        n_voxels = np.prod(brain_shape)

        # ---- 用于存放 5 折的测试数据和预测数据（在掩码内的voxel）----
        testY_all = []
        predY_all = []

        # ==================== 交叉验证循环 ====================
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]

            # 取训练集 BOLD: (x, y, z, train_t)
            train_bold_data = bold_data[..., train_idx]
            # 先展平空间，再取转置 -> (train_t, x*y*z)
            y_train = train_bold_data.reshape(-1, len(train_idx)).T  # (train_t, n_voxels)

            # 对应地，测试集 BOLD
            test_bold_data = bold_data[..., test_idx]
            y_test = test_bold_data.reshape(-1, len(test_idx)).T  # (test_t, n_voxels)

            # ========== 应用语言掩码 (只保留语言区 voxel) ==========
            mask_1d = language_mask.ravel()
            y_train = y_train[:, mask_1d]  # (train_t, masked_voxels)
            y_test  = y_test[:,  mask_1d]  # (test_t,  masked_voxels)

            # ---------- 标准化 X ----------
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled  = X_scaler.transform(X_test)

            # ---------- 标准化 Y (多输出) ----------
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled  = y_scaler.transform(y_test)

            # ---------- 训练模型 ----------
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            model.fit(X_train_scaled, y_train_scaled)

            # ---------- 预测 ----------
            y_pred_scaled = model.predict(X_test_scaled)

            # 收集所有 fold 的 测试真值/预测值 (标准化后)
            testY_all.append(y_test_scaled)
            predY_all.append(y_pred_scaled)

        # ==================== 拼接所有 fold 的测试集 ====================
        testY_all = np.concatenate(testY_all, axis=0)  # (total_test_samples, masked_voxels)
        predY_all = np.concatenate(predY_all, axis=0) # (total_test_samples, masked_voxels)

        # ========== 手动公式，计算 Pearson 相关系数 (逐 voxel) ==========
        testY_mean = np.mean(testY_all, axis=0, keepdims=True)
        predY_mean = np.mean(predY_all, axis=0, keepdims=True)

        numerator = np.sum((testY_all - testY_mean) * (predY_all - predY_mean), axis=0)
        denominator = np.sqrt(
            np.sum((testY_all - testY_mean)**2, axis=0) *
            np.sum((predY_all - predY_mean)**2, axis=0)
        ) + 1e-12

        r_per_voxel_masked_manual = numerator / denominator  # shape: (masked_voxels,)

        # ========== 将 手动公式 的 r 放回到 3D (非语言区为 NaN) ==========
        r_3d_manual = np.full(brain_shape, np.nan, dtype=np.float32)
        r_3d_manual[language_mask] = r_per_voxel_masked_manual

        # 计算语言区平均相关(手动公式)
        language_region_r_manual = np.nanmean(r_3d_manual)

        # ========== DEBUG: Compare with pearsonr (同第一段脚本) ==========
        from scipy.stats import pearsonr
        r_per_voxel_masked_pearson = np.full(testY_all.shape[1], np.nan, dtype=np.float32)
        p_per_voxel_masked_pearson = np.full(testY_all.shape[1], np.nan, dtype=np.float32)

        for i in range(testY_all.shape[1]):
            r_val, p_val = pearsonr(testY_all[:, i], predY_all[:, i])
            r_per_voxel_masked_pearson[i] = r_val
            p_per_voxel_masked_pearson[i] = p_val

        # 放回3D
        r_3d_pearson = np.full(brain_shape, np.nan, dtype=np.float32)
        r_3d_pearson[language_mask] = r_per_voxel_masked_pearson
        language_region_r_pearson = np.nanmean(r_3d_pearson)

        # ========== 打印对比信息：手动公式 vs pearsonr ==========
        print(f"{sub_id} - 手动公式平均 r: {language_region_r_manual:.5f},  pearsonr 平均 r: {language_region_r_pearson:.5f}")

        # 对比逐 voxel 差异
        r_diff = r_per_voxel_masked_manual - r_per_voxel_masked_pearson
        max_diff = np.nanmax(np.abs(r_diff))
        mean_diff = np.nanmean(np.abs(r_diff))
        print(f"{sub_id} - voxel级别最大绝对差: {max_diff:.6f}, 平均绝对差: {mean_diff:.6f}\n")

        # ---------- 记录结果 ----------
        results.append({
            'subject': sub_id,
            'mean_pearson_r_manual': language_region_r_manual,
            'mean_pearson_r_scipy': language_region_r_pearson,
            'max_voxel_diff': max_diff,
            'mean_voxel_diff': mean_diff,
            'total_voxels': n_voxels,
            'masked_voxels': np.sum(language_mask),
            'valid_voxels': np.sum(~np.isnan(r_3d_manual))
        })

        # 若需要输出 NIfTI
        if args.output_nii:
            output_img_manual = nib.Nifti1Image(r_3d_manual, bold_img.affine)
            out_name_manual = f"{sub_id}_{hrf_model}_r_manual.nii.gz"
            output_path_manual = os.path.join(args.output_dir, out_name_manual)
            nib.save(output_img_manual, output_path_manual)

            output_img_pearson = nib.Nifti1Image(r_3d_pearson, bold_img.affine)
            out_name_pearson = f"{sub_id}_{hrf_model}_r_pearson.nii.gz"
            output_path_pearson = os.path.join(args.output_dir, out_name_pearson)
            nib.save(output_img_pearson, output_path_pearson)

    # ========== 汇总结果 =============
    if results:
        df = pd.DataFrame(results)
        # 计算整体平均(手动)与(pearson)
        grand_avg_manual = df['mean_pearson_r_manual'].mean()
        grand_avg_scipy = df['mean_pearson_r_scipy'].mean()

        df = pd.concat([
            df,
            pd.DataFrame([{
                'subject': 'average',
                'mean_pearson_r_manual': grand_avg_manual,
                'mean_pearson_r_scipy': grand_avg_scipy,
                'max_voxel_diff': np.nan,
                'mean_voxel_diff': np.nan,
                'total_voxels': np.nan,
                'masked_voxels': np.nan,
                'valid_voxels': np.nan
            }])
        ], ignore_index=True)
        
        output_file = os.path.join(args.output_dir, f"{hrf_model}_{lang}_{args.alpha}_langMasked_compare.csv")
        df.to_csv(output_file, index=False)
        print(f"\n对比结果保存至: {output_file}")
    else:
        print("\n未处理任何被试数据")


if __name__ == "__main__":
    main()
