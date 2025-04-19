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
from scipy.stats import pearsonr
import warnings

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
    print(subjects)
    print(args.hrf_csv)
    
    if not subjects:
        raise ValueError(f"未找到{lang}语言的被试数据")

    # 加载设计矩阵 (X)
    X = pd.read_csv(args.hrf_csv, header=None).values  # shape: (n_timepoints, n_features)
    n_timepoints = X.shape[0]

    # 加载语言概率图，并创建二值掩码
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
    
    #temp = [subjects[0]]
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

        # 用于存放 5 折的测试数据和预测数据（在掩码内的voxel）
        testY_all = []
        predY_all = []

        # 交叉验证循环
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]

            # 取训练集 BOLD: (x, y, z, train_t)
            train_bold_data = bold_data[..., train_idx]
            # 先展平空间，再取转置 -> (train_t, x*y*z)
            y_train = train_bold_data.reshape(-1, len(train_idx)).T  # (train_t, n_voxels)

            # 对应地，测试集 BOLD
            test_bold_data = bold_data[..., test_idx]
            y_test = test_bold_data.reshape(-1, len(test_idx)).T  # (test_t, n_voxels)

            # 应用语言掩码 (只保留语言区 voxel)
            mask_1d = language_mask.ravel()
            y_train = y_train[:, mask_1d]  # (train_t, masked_voxels)
            y_test  = y_test[:,  mask_1d]  # (test_t,  masked_voxels)

            # 标准化 X
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled  = X_scaler.transform(X_test)

            # 标准化 Y (多输出)
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled  = y_scaler.transform(y_test)

            # 训练模型
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            model.fit(X_train_scaled, y_train_scaled)

            # 预测
            y_pred_scaled = model.predict(X_test_scaled)

            # 收集所有 fold 的 测试真值/预测值 (标准化后)
            testY_all.append(y_test_scaled)
            predY_all.append(y_pred_scaled)


        # 拼接所有 fold 的测试集
        testY_all = np.concatenate(testY_all, axis=0)  # (total_test_samples, masked_voxels)
        predY_all = np.concatenate(predY_all, axis=0)  # (total_test_samples, masked_voxels)
        #print("testY_all", testY_all.shape) (2816, 25137)




        # 使用 pearsonr 计算相关系数，避免常量输入警告
        r_per_voxel_masked = np.full(testY_all.shape[1], np.nan, dtype=np.float32)
        p_per_voxel_masked = np.full(testY_all.shape[1], np.nan, dtype=np.float32)
        
        # 跟踪统计信息
        constant_voxels = 0
        tiny_p_values = 0
        #p_value_bins = np.zeros(6)  # [0-0.0001, 0.0001-0.001, 0.001-0.01, 0.01-0.05, 0.05-0.1, >0.1]
        
        # 临时禁用警告以避免大量输出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            for i in range(testY_all.shape[1]):
                test_data = testY_all[:, i]
                pred_data = predY_all[:, i]
                
                # 检查输入是否为常量（所有值相同）
                if np.all(test_data == test_data[0]) or np.all(pred_data == pred_data[0]):
                    constant_voxels += 1
                    r_per_voxel_masked[i] = np.nan
                    p_per_voxel_masked[i] = np.nan
                    continue
                
                # 计算相关系数
                r_val, p_val = pearsonr(test_data, pred_data)
                r_per_voxel_masked[i] = r_val
                p_per_voxel_masked[i] = p_val
                #print(p_val)
                

        # 放回3D
        r_3d = np.full(brain_shape, np.nan, dtype=np.float32)
        p_3d = np.full(brain_shape, np.nan, dtype=np.float32)
        r_3d[language_mask] = r_per_voxel_masked
        p_3d[language_mask] = p_per_voxel_masked
        
        # 计算语言区平均相关系数
        language_region_r = np.nanmean(r_3d)
        
        # 统计p值
        valid_p_values = p_per_voxel_masked[~np.isnan(p_per_voxel_masked)]
        #min_p = np.min(valid_p_values) if len(valid_p_values) > 0 else np.nan
        mean_p = np.mean(valid_p_values) if len(valid_p_values) > 0 else np.nan
        #median_p = np.median(valid_p_values) if len(valid_p_values) > 0 else np.nan
        
        # 打印结果和调试信息
        print(f"\n{sub_id} - 分析结果：")
        print(f"  Pearson平均 r: {language_region_r:.6f}")
        print(f"  非NaN的voxel数量: {np.sum(~np.isnan(r_3d))}")
        print(f"  常量voxel数量: {constant_voxels} (已跳过)")
        
        # P值调试信息
        #print(f"\nP值分析：")
        #print(f"  P值统计： 平均={mean_p:.10f}")
        #print(f"  极小P值 (< 0.0001) 数量: {tiny_p_values} ({tiny_p_values/max(1, len(valid_p_values))*100:.2f}%)")
        
        # P值分布
        # p_bins_labels = ["<0.0001", "0.0001-0.001", "0.001-0.01", "0.01-0.05", "0.05-0.1", ">0.1"]
        # for i, count in enumerate(p_value_bins):
        #     percentage = count / max(1, len(valid_p_values)) * 100
        #     print(f"  P值 {p_bins_labels[i]}: {count:.0f} ({percentage:.2f}%)")
        
        # 记录结果
        results.append({
            'subject': sub_id,
            'mean_pearson_r': language_region_r,
            'mean_p_value': mean_p, 
            'tiny_p_values_pct': tiny_p_values/max(1, len(valid_p_values))*100,
            'constant_voxels': constant_voxels,
            'total_voxels': n_voxels,
            'masked_voxels': np.sum(language_mask),
            'valid_voxels': np.sum(~np.isnan(r_3d))
        })

        # 若需要输出 NIfTI
        if args.output_nii:
            # 保存R值图
            output_img_r = nib.Nifti1Image(r_3d, bold_img.affine)
            out_name_r = f"{sub_id}_{hrf_model}_r.nii.gz"
            output_path_r = os.path.join(args.output_dir, out_name_r)
            nib.save(output_img_r, output_path_r)
            
            # 保存显著性图（负对数p值，更适合可视化）
            # 转换p值到-log10(p)以便可视化
            # 注意：较大的值表示更高的显著性
            neg_log_p = -np.log10(p_3d)
            neg_log_p[np.isinf(neg_log_p)] = np.nan  # 处理p=0导致的无穷大
            
            output_img_p = nib.Nifti1Image(neg_log_p, bold_img.affine)
            out_name_p = f"{sub_id}_{hrf_model}_neglog10p.nii.gz"
            output_path_p = os.path.join(args.output_dir, out_name_p)
            nib.save(output_img_p, output_path_p)

    # 汇总结果
    if results:
        df = pd.DataFrame(results)
        # 计算整体平均
        grand_avg_r = df['mean_pearson_r'].mean()
        grand_avg_p = df['mean_p_value'].mean() if 'mean_p_value' in df else np.nan

        df = pd.concat([
            df,
            pd.DataFrame([{
                'subject': 'average',
                'mean_pearson_r': grand_avg_r,
                'mean_p_value': grand_avg_p,
                'tiny_p_values_pct': np.nan,
                'constant_voxels': np.nan,
                'total_voxels': np.nan,
                'masked_voxels': np.nan,
                'valid_voxels': np.nan
            }])
        ], ignore_index=True)
        
        output_file = os.path.join(args.output_dir, f"{hrf_model}_{lang}_{args.alpha}_langMasked.csv")
        df.to_csv(output_file, index=False)
        print(f"\n结果保存至: {output_file}")
    else:
        print("\n未处理任何被试数据")


if __name__ == "__main__":
    main()