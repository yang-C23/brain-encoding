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

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_model_Ridge(X, y, n, saveto, save_results = True, shuffle=False, prefix = ""):
    if shuffle: # note that shuffling might artificially increase the encoding scores. Default is non-shuffled. All the analyses now are w/o shuffling.
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    # 用来存储每折的相关系数(r)
    out_coefs = []
    # 用来存储每折学到的回归系数（长度应为 300，因为特征数是 300)
    out_pred = []; y_tot = []
    # 用来收集所有折的预测值（测试集上的预测），最终合并一起算整体 Pearson 相关。 收集所有折的真实值。
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # X_scaler / y_scaler: 分别用于对 X 和 y 做 StandardScaler(均值为 0、标准差为 1)的变换对象。
    for train_index, test_index in tqdm(kf.split(X), total=n):
        # Normalizing embedding and responses. The normalization parameters are always estimated on the training data and transferred to the test data
        X_train = X_scaler.fit_transform(X[train_index])
        # X[train_index]: ({train_size}, 300)
        # 调用 fit_transform 会： 先根据 X_train 的均值和标准差做拟合。 再对 X_train 做标准化（每个特征减去均值，除以标准差）。
        # X_train 这里用的是同样的均值、标准差（来自训练集的fit），形状 ({test_size}, 300)。
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        # y[train_index]: 原本是 ({train_size},)，通过 reshape(-1,1) 变成 ({train_size}, 1)，方便 StandardScaler 一次性处理单列数据
        # 最终 y_train 和 y_test 都是一维向量，形状分别为 ({train_size},) 和 ({test_size},)
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
        reg = RidgeCV(alphas=0.1)
        #reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)) # Alpha values log spaced. Alpha chosen with leave-one-out nested CV
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        #({test_size},)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        # 本折的预测结果合并进一个全局列表 out_pred
        y_tot.extend(y_test.tolist())
        # 把真实值也存进一个全局列表 y_tot
        coefs = reg.coef_#; print(coefs)
        # 回归系数 (300,)，存到 out_coefs 里
        out_coefs.append(coefs)
        out_reg.append(r)
        #把该折的 Pearson 𝑟 放进列表
    #print(round(np.mean(out_reg), 4))
    r_tot = pearsonr(out_pred, y_tot)[0]
    print(round(r_tot, 4))
    out_predictions = [y_tot, out_pred]
    if save_results:
        save(r_tot, f"results/rs/{prefix}{saveto}")
        save(out_reg, f"results/out_reg/{prefix}{saveto}") # saving all rs and coefficients for later use
        save(out_coefs, f"results/coefficients/{prefix}{saveto}")
        save(out_predictions, f"results/predictions/{prefix}{saveto}")
    return r_tot

# K折划分：把该 voxel 对应的 BOLD 信号 𝑦
# y 分成训练集和测试集。
# 对 X 和 y 均做标准化（StandardScaler），然后用 RidgeCV 拟合。
# 把每折测试集上的预测拼接起来(out_pred)和真实值(y_tot)拼接起来。
# 在所有折全部跑完后，直接用 pearsonr(out_pred, y_tot) 得到一个整体的相关系数。


def main():
    parser = argparse.ArgumentParser(description='全脑编码分析')
    parser.add_argument('--hrf_csv', type=str, required=True,
                      help='HRF模型文件路径')
    parser.add_argument('--derivatives_dir', type=str, required=True,
                      help='BOLD数据根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='结果输出目录')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='KFold折数, 默认为5')
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
    X = pd.read_csv(args.hrf_csv, header=None).values
    n_timepoints = X.shape[0]


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
        r_map = np.zeros(brain_shape, dtype=np.float32) * np.nan  # 初始化 NaN

        # 将 BOLD reshape 为 (n_voxels, T)
        n_voxels = np.prod(brain_shape)
        bold_2d = bold_data.reshape(n_voxels, n_timepoints)

        # (5) 循环每个voxel, 调用 test_model_Ridge(X, y, n_splits, ...)
        #    y就是该voxel的时间序列
        for vox_idx in tqdm(range(n_voxels), desc="voxel-wise Ridge"):
            y = bold_2d[vox_idx, :]  # shape (2816, )
            print("voxel", vox_idx, " std of y =", np.std(y))

            
            # 调用 test_model_Ridge 
            #   - n_splits=args.n_splits
            #   - 这里我们传一个文件名可随意, 假设"voxelXXXX"之类, 也可不保存
            #   - save_results=False 避免产生大量文件
            r_tot = test_model_Ridge(X, y, args.n_splits, 
                                    saveto=f"voxel_{vox_idx}", 
                                    save_results=False, 
                                    shuffle=False)


            # 将这个voxel的相关值存到 r_map 里
            r_map.flat[vox_idx] = r_tot
        
        # (6) 全脑平均r值
        #     注意：如果某些voxel值是NaN或无效，可以用 ~np.isnan(r_map) 做筛选
        brain_mean_r = np.nanmean(r_map)
        print(sub_id,", Whole brain average r:", brain_mean_r)
        results.append(brain_mean_r)


        if args.output_nii:
            out_nii = nib.Nifti1Image(r_map, affine=bold_img.affine)
            out_path = os.path.join(args.output_dir, f"{sub_id}_voxelwise_r.nii.gz")
            nib.save(out_nii, out_path)
            print("Saved voxelwise r-map to:", out_path)

        csv_path = os.path.join(args.output_dir, f"{sub_id}_rmap_summary.csv")
        pd.DataFrame({"voxel_mean_r": [brain_mean_r]}).to_csv(csv_path, index=False)


    # 生成汇总结果
    if results:
        df = pd.DataFrame(results, columns=["mean_pearson_r"])
        df.insert(0, "subject", subjects)  # 添加被试 ID 列
        grand_avg = df["mean_pearson_r"].mean()

        # 添加整体平均值
        df = pd.concat([df, pd.DataFrame([{
            "subject": "average",
            "mean_pearson_r": grand_avg,
        }])], ignore_index=True)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{hrf_model}_{lang}_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\n分析完成，结果保存至: {output_file}")
    else:
        print("\n未处理任何被试数据")


if __name__ == "__main__":
    main()