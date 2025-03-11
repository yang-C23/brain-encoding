import os
import argparse
import glob

import numpy as np
import pandas as pd
import nibabel as nib

from tqdm import tqdm

# rsatoolbox 用于计算 RDM
import rsatoolbox

def parse_args():
    parser = argparse.ArgumentParser(description="对比模型HRF与大脑BOLD的RDM")
    parser.add_argument("--hrf_csv", type=str, required=True,
                        help="模型 HRF CSV 文件路径 (无表头, header=None)")
    parser.add_argument("--derivatives_dir", type=str, required=True,
                        help="BOLD数据根目录 (内含 sub-xxx/func 等文件夹)")
    parser.add_argument("--lang_atlas", type=str, required=True,
                        help="语言概率图 (SPM_LanA_n806.nii) 的路径")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="语言图阈值，若 voxel 的概率 >= threshold 则视为语言区")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="结果输出目录")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    hrf_model = os.path.basename(args.hrf_csv).split('_')[0]

    # ============ 1) 读取并计算「模型 HRF」的 RDM ============
    print(f"[INFO] 读取模型 HRF CSV: {args.hrf_csv}")
    df_model = pd.read_csv(args.hrf_csv, header=None)  # shape: [N, K]
    data_model = df_model.values

    # （可选）去除标准差为 0 的时间点，防止后面距离为0
    # ----------------------------------------------
    # stds_model = np.std(data_model, axis=1)
    # zero_std_mask = (stds_model == 0)
    # if np.any(zero_std_mask):
    #     print(f"[WARNING] 模型HRF有 {np.sum(zero_std_mask)} 个时间点标准差为0，将被剔除。")
    # data_model = data_model[~zero_std_mask]

    # 使用 rsatoolbox 计算 RDM
    print("data_model",data_model.shape)
    d_model = rsatoolbox.data.Dataset(data_model)   # 默认: observations×features
    rdm_model = rsatoolbox.rdm.calc_rdm(d_model)    # 得到一组(或一个) RDM 对象
    print("rdm_model", rdm_model)


    # ============ 2) 加载语言区图，并创建掩码 ============
    atlas_img = nib.load(args.lang_atlas)
    atlas_data = atlas_img.get_fdata()  # shape: (X, Y, Z)
    if atlas_data.ndim != 3:
        raise ValueError("语言图应该是3D图，请检查文件。")

    language_mask = (atlas_data >= args.threshold)

    # ============ 3) 遍历所有被试，计算大脑 RDM 并与模型 RDM 做相关 ============
    # 从路径提取语言代码
    lang = os.path.normpath(args.hrf_csv).split(os.sep)[-2]

    # 查找匹配的被试
    subject_dirs = glob.glob(os.path.join(args.derivatives_dir, f'sub-{lang}*'))
    subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
    
    if not subjects:
        raise ValueError(f"未找到{lang}语言的被试数据")

    # # 加载设计矩阵 (X)
    # X = pd.read_csv(args.hrf_csv, header=None).values  # shape: (n_timepoints, n_features)
    # n_timepoints = X.shape[0]


    results = []
    for sub_id in tqdm(subjects, desc="处理被试"):

        bold_pattern = os.path.join(
            args.derivatives_dir, sub_id, 'func',
            f'{sub_id}_task-lpp*_mergedTP-*_bold.nii.gz'
        )


        bold_files = glob.glob(bold_pattern)
        if len(bold_files) == 0:
            print(f"  跳过 {sub_id}: 未找到匹配的 BOLD 文件, pattern={args.bold_pattern}")
            continue

        # 这里只示范读取第一个匹配到的文件
        bold_path = bold_files[0]
        try:
            bold_img = nib.load(bold_path)
            bold_data = bold_img.get_fdata()  # shape: (X, Y, Z, T)
        except Exception as e:
            print(f"  跳过 {sub_id}: 加载BOLD失败: {e}")
            continue

        # 检查 BOLD 与 atlas 的空间维度是否匹配
        if bold_data.shape[:3] != atlas_data.shape:
            print(f"  跳过 {sub_id}: BOLD与语言图空间维度不匹配")
            continue

        n_timepoints = bold_data.shape[-1]
        brain_shape = bold_data.shape[:3]
        print("bold_data", bold_data.shape)

        # 3D -> 2D (T × voxels)，然后再应用语言区掩码
        # 注意：BOLD_data 是 (X, Y, Z, T)，转置后得到 (T, X, Y, Z)，再压平空间维度
        bold_2d = bold_data.reshape(-1, n_timepoints).T  # shape: (T, X*Y*Z)
        print("bold_2d", bold_2d)
        mask_1d = language_mask.ravel()

        bold_masked = bold_2d[:, mask_1d]               # shape: (T, masked_voxels)


        print("bold_masked", bold_masked)
        # 如果需要去除标准差为 0 的时间点，可以类似上面对 model HRF 的处理
        # stds_bold = np.std(bold_masked, axis=1)
        # zero_std_mask_bold = (stds_bold == 0)
        # if np.any(zero_std_mask_bold):
        #     print(f"  [WARNING] {sub_id} 有 {np.sum(zero_std_mask_bold)} 个时间点在语言区内方差为0，将被剔除。")
        # bold_masked = bold_masked[~zero_std_mask_bold]

        # # 如果剔除后 timepoints 数量太少，也就无法计算RDM，跳过
        # if bold_masked.shape[0] < 2:
        #     print(f"  跳过 {sub_id}: 去除后时间点数量不足以计算RDM")
        #     continue

        # 生成一个 rsatoolbox Dataset，observations=时间点，features=voxel
        d_brain = rsatoolbox.data.Dataset(bold_masked) 
        rdm_brain = rsatoolbox.rdm.calc_rdm(d_brain)  # 大脑 RDM
        
        print("rdm_brain", rdm_brain)

        # ============ 4) 比较 RDM ============
        # rsatoolbox.rdm.compare 的方法有多种，常用 'corr', 'cosine', 'spearman' 等
        # 这里以 Pearson 相关为例:
        rdm_corr = rsatoolbox.rdm.compare(rdm_brain, rdm_model, method='corr')
        # 注意 compare() 返回的是一个数组(如果两个 RDM 对象都只有1个RDM)，则一般是长度为1的数组
        # 所以我们取 [0] 取出具体相关值
        print(rdm_corr.shape)
        corr_val = float(rdm_corr[0])  # 如果 rdm_corr.shape == (1,)
        print(f"{sub_id} 与模型 RDM 的相关: {corr_val:.4f}")


        results.append({
            "subject": sub_id,
            "n_timepoints_original": bold_data.shape[-1],
            "n_timepoints_used": bold_masked.shape[0],
            "rdm_correlation": corr_val
        })

    # ============ 5) 保存结果 ============
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        # 计算所有被试平均
        mean_corr = df_results["rdm_correlation"].mean()
        df_results.loc[len(df_results)] = {
            "subject": "average",
            "n_timepoints_original": np.nan,
            "n_timepoints_used": np.nan,
            "rdm_correlation": mean_corr
        }

        out_csv = os.path.join(args.output_dir, f"{hrf_model}_RDM.csv")
        df_results.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\n[INFO] 完成！结果已保存到: {out_csv}")
    else:
        print("\n[INFO] 未有任何被试结果可保存。")


if __name__ == "__main__":
    main()
