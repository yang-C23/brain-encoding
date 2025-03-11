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




def main():
    parser = argparse.ArgumentParser(description='全脑编码分析')
    parser.add_argument('--hrf_csv', type=str, required=True,
                      help='HRF模型文件路径')
    parser.add_argument('--derivatives_dir', type=str, required=True,
                      help='BOLD数据根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='结果输出目录')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='Ridge回归的正则化系数')
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

    # 加载设计矩阵
    X = pd.read_csv(args.hrf_csv, header=None).values#(2816,n features)
    n_timepoints = X.shape[0] #(2816)


    # 现在替换为高斯噪声，同样保留时间点数量
    # 为了演示，先读取文件，仅获取形状信息：
    # X_shape = pd.read_csv(args.hrf_csv, header=None).values.shape
    # n_timepoints = X_shape[0]

    # 生成与原 HRF 相同形状的高斯噪声
    # loc=0, scale=1 为均值 0、标准差 1，可根据需求调整
    # X = np.random.normal(loc=0.0, scale=1.0, size=X_shape)

    results = []


    for sub_id in tqdm(subjects, desc='处理被试'):
        # 查找BOLD文件
        # bold_pattern = os.path.join(args.derivatives_dir, sub_id, 'func',
        #                             f'{sub_id}_task-lpp{lang}_mergedTP-*_bold.nii.gz')
        bold_pattern = os.path.join(args.derivatives_dir, sub_id, 'func',
                                    f'{sub_id}_task-lpp*_mergedTP-*_bold.nii.gz')
        bold_files = glob.glob(bold_pattern)
        
        if not bold_files:
            print(f"\n跳过{sub_id}: 未找到BOLD文件")
            continue
            
        try:
            bold_img = nib.load(bold_files[0])
            bold_data = bold_img.get_fdata()
            #(73, 90, 74, 2816)
        except Exception as e:
            print(f"\n{sub_id}加载BOLD数据失败: {str(e)}")
            continue

        # 验证时间维度
        if bold_data.shape[-1] != n_timepoints:
            print(f"\n跳过{sub_id}: 时间点不匹配")
            continue

        # 准备全脑数据
        brain_shape = bold_data.shape[:-1]# (73, 90, 74)
        n_voxels = np.prod(brain_shape)  # 73*90*74 = 485180
        
        # 初始化交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r = []  # 用于存放每折的相关系数结果（3D）
        all_r_folds = []
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        testY_all = []
        predY_all = []

        for train_idx, test_idx in kf.split(X):
            # 训练集与测试集
            X_train, X_test = X[train_idx], X[test_idx] #(train t_point, n featrures)
            
            # 重塑BOLD数据为 (时间点, 体素)
            y_train = bold_data[..., train_idx].reshape(-1, len(train_idx)).T 
            #(73, 90, 74, train t_point)
            #(train t_point, 485,180)

            y_test = bold_data[..., test_idx].reshape(-1, len(test_idx)).T

            # 2) 对 X 做标准化
            X_train_scaled = X_scaler.fit_transform(X_train)  # (train_size, n_features)
            X_test_scaled  = X_scaler.transform(X_test)       # (test_size,  n_features)

            #  由于 y 是多输出 (train_size, n_voxels)，StandardScaler 会对每个 voxel 独立估计 mean/std
            y_train_scaled = y_scaler.fit_transform(y_train)  # (train_size, n_voxels)
            y_test_scaled  = y_scaler.transform(y_test)       # (test_size,  n_voxels)

            # 训练模型
            #model = Ridge(alpha=args.alpha)
            #model = RidgeCV(alphas=(0.01,0.1,1,10)), cv=3)
            # model = make_pipeline(
            #     StandardScaler(with_std=True),  # 中心化+标准化
            #     Ridge(alpha=args.alpha))
            # model.fit(X_train, y_train)

            model = RidgeCV(alphas=0.1)
            model.fit(X_train_scaled, y_train_scaled)
            
            # 预测
            # y_pred = model.predict(X_test)
            y_pred_scaled = model.predict(X_test_scaled)

            # 收集所有 fold 的测试真值/预测值 (标准化后的)
            testY_all.append(y_test_scaled) 
            predY_all.append(y_pred_scaled)
        
        # ---------- 6) 拼接所有 fold 的测试真值与预测值 ----------
        testY_all = np.concatenate(testY_all, axis=0)  # (total_test_samples, n_voxels)
        predY_all = np.concatenate(predY_all, axis=0)

        # ---------- 7) 计算合并后整体的 Pearson 相关系数 ----------
        testY_mean = np.mean(testY_all, axis=0, keepdims=True) 
        predY_mean = np.mean(predY_all, axis=0, keepdims=True)

        
        # all_r_folds = np.array(all_r_folds)  # shape: (5, n_voxels)
        # mean_r = np.mean(all_r_folds, axis=0)   # (n_voxels,) 每个voxel的平均相关
        # roi_mean = np.nanmean(mean_r)           # 再对体素做平均
        # print("ROI or Whole-brain average r:", roi_mean)


        numerator = np.sum((testY_all - testY_mean) * (predY_all - predY_mean), axis=0)
        denominator = np.sqrt(
            np.sum((testY_all - testY_mean)**2, axis=0) *
            np.sum((predY_all - predY_mean)**2, axis=0)
        ) + 1e-12  # 避免除零

        r_per_voxel = numerator / denominator  # (n_voxels, )

        # 将其重塑回 3D 空间，用于输出 NIfTI
        r_3d = r_per_voxel.reshape(brain_shape)

        # 计算全脑平均
        whole_brain_r = np.nanmean(r_per_voxel)
        print(f"{sub_id} - Whole-brain average r: {whole_brain_r}")

        # ---------- 8) 收集结果 ----------
        results.append({
            'subject': sub_id,
            'mean_pearson_r': whole_brain_r,
            'total_voxels': n_voxels,
            'valid_voxels': np.sum(~np.isnan(r_3d))
        })

        # 若需要输出 NIfTI
        if args.output_nii:
            output_img = nib.Nifti1Image(r_3d, bold_img.affine)
            output_path = os.path.join(args.output_dir, f"{sub_id}_{hrf_model}_pearson_r.nii.gz")
            nib.save(output_img, output_path)


        # # 计算交叉验证后，取所有折的平均相关系数
        # mean_r = np.nanmean(cv_r, axis=0)  # 大小仍为 3D
        # sub_mean = np.nanmean(mean_r)      # 整个脑区的平均相关系数
        # # 同样的K折 拟合，但这里一次性拟合所有voxel（或说把所有voxel放在y_train 的不同列），最后得到少。每个 voxel都会在同一个模型下得到自己的预测序列g。
        # # 计算每折内「真实与预测」的 Pearson 个，再把对所有折的， 做平均；或者有时也会先把每折的预测拼接在一起，再算一个大测试集的相关。
        # results.append({
        #     'subject': sub_id,
        #     'mean_pearson_r': sub_mean,
        #     'total_voxels': n_voxels,
        #     'valid_voxels': np.sum(~np.isnan(mean_r))
        # })

        # if args.output_nii:
        #     output_img = nib.Nifti1Image(mean_r, bold_img.affine)
        #     output_path = os.path.join(args.output_dir, f"{sub_id}_{hrf_model}_pearson_r.nii.gz")
        #     nib.save(output_img, output_path)

    # 生成汇总结果
    if results:
        df = pd.DataFrame(results)
        grand_avg = df['mean_pearson_r'].mean()
        df = pd.concat([df, pd.DataFrame([{
            'subject': 'average',
            'mean_pearson_r': grand_avg,
            'total_voxels': np.nan,
            'valid_voxels': np.nan
        }])], ignore_index=True)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{hrf_model}_{lang}_{args.alpha}_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\n分析完成，结果保存至: {output_file}")
    else:
        print("\n未处理任何被试数据")

if __name__ == "__main__":
    main()