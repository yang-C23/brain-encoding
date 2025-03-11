#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy import spatial
import numpy, rsatoolbox


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and compare RDM of two HRF-convolved CSVs from event2hrf.py.")
    parser.add_argument("--hrf_csv1", type=str, required=True,
                        help="First CSV path (output from event2hrf.py, header=None).")
    parser.add_argument("--hrf_csv2", type=str, required=True,
                        help="Second CSV path (output from event2hrf.py, header=None).")
    parser.add_argument("--output_result", type=str, default="rdm_compare.npz",
                        help="Output npz file containing rdm1, rdm2, and rdm_correlation.")
    return parser.parse_args()

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
            #rdm[i, j] = corr
    return rdm

def rdm_correlation(rdm1, rdm2):
    """
    计算两个 RDM (N x N) 的相关系数（仅上三角部分).
    """
    N = rdm1.shape[0]
    print(N)
    # 取上三角 (i < j) 的索引
    iu = np.triu_indices(N, k=1)

    rdm1_vals = rdm1[iu]
    rdm2_vals = rdm2[iu]
    return np.corrcoef(rdm1_vals, rdm2_vals)[0, 1]

def main():
    args = parse_args()

    # 1) 读取两个 HRF CSV (都无表头, header=None)
    #    event2hrf.py 生成形状 ~ [n_scans, ncol] 的时序矩阵
    print(f"Loading HRF CSV1 from: {args.hrf_csv1}")
    df1 = pd.read_csv(args.hrf_csv1, header=None)
    data1 = df1.values  # shape [N, K1]

    print(f"Loading HRF CSV2 from: {args.hrf_csv2}")
    df2 = pd.read_csv(args.hrf_csv2, header=None)
    data2 = df2.values  # shape [N, K2]

    # 2) 对两个 HRF 矩阵的行数进行检查
    #    行数必须相同 (相同 n_scans), 否则没法做一一对应的 RDM 对比
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(
            f"Mismatch in timepoints (rows). "
            f"hrf_csv1 has {data1.shape[0]} rows, while hrf_csv2 has {data2.shape[0]} rows."
        )




    # 2) 计算每行的标准差
    stds1 = np.std(data1, axis=1)

    # 3) 找出标准差为 0 的行（这些行全为常数或全0）
    zero_std_mask1 = (stds1 == 0)
    removed_timepoints = np.where(zero_std_mask1)[0]  # 记录被剔除的行索引
    num_removed1 = len(removed_timepoints)

    # 4) 剔除这些行
    filtered_data1 = data1[~zero_std_mask1]
    N_filtered1 = filtered_data1.shape[0]

    print(f"Found {num_removed1} rows with zero std (out of ).")
    print(f"After removal, we have {N_filtered1} rows.")


    # 2) 计算每行的标准差
    stds2 = np.std(data2, axis=1)

    # 3) 找出标准差为 0 的行（这些行全为常数或全0）
    zero_std_mask2 = (stds2 == 0)
    removed_timepoints = np.where(zero_std_mask2)[0]  # 记录被剔除的行索引
    num_removed2 = len(removed_timepoints)

    # 4) 剔除这些行
    filtered_data2 = data2[~zero_std_mask2]
    N_filtered2 = filtered_data1.shape[0]

    print(f"Found {num_removed2} rows with zero std (out of ).")
    print(f"After removal, we have {N_filtered2} rows.")

    

    # 3) 分别计算 RDM
    # print("Computing RDM1...")
    # rdm1 = compute_rdm(filtered_data1)  # [N, N]
    # print("Computing RDM2...")
    # rdm2 = compute_rdm(filtered_data2)  # [N, N]



    # 4) 保存 RDM 到 CSV (可选，也可改为 npy)
    # print(f"Saving RDM1 to {args.output_rdm1}, shape={rdm1.shape}")
    # np.savetxt(args.output_rdm1, rdm1, delimiter=",")
    # print(f"Saving RDM2 to {args.output_rdm2}, shape={rdm2.shape}")
    # np.savetxt(args.output_rdm2, rdm2, delimiter=",")


    # # 5) 计算两个 RDM 的相关系数
    # corr = rdm_correlation(rdm1, rdm2)
    # print("=== RESULTS ===")
    # print(f"RDM correlation between CSV1 and CSV2: {corr:.4f}")

    d1 = rsatoolbox.data.Dataset(filtered_data1)
    d2 = rsatoolbox.data.Dataset(filtered_data2)
    print(filtered_data2.shape)
    rdms1 = rsatoolbox.rdm.calc_rdm(d1)
    rdms2 = rsatoolbox.rdm.calc_rdm(d2)
    #corr = rsatoolbox.rdm.compare.compare_correlation(rdms1,rdms2)
    corr = rsatoolbox.rdm.compare(rdms1,rdms2,method='corr')
    print("RDM corr", corr)

    #N_filtered2 = filtered_data2.shape[0]
    cosins = np.zeros(N_filtered2)
    # print(cosins.shape)
    pearsons = np.zeros(N_filtered2)
    for i in range(N_filtered2):
        data1_flat = filtered_data1[i].flatten()
        data2_flat = filtered_data2[i].flatten()
        cos_sim = 1 - spatial.distance.cosine(data1_flat, data2_flat)
        cosins[i] = cos_sim
        pearson_corr = np.corrcoef(data1_flat, data2_flat)[0, 1]
        pearsons[i] = pearson_corr

    # data1_flat = data1[2].flatten()
    # data2_flat = data2[2].flatten()
    # print(spatial.distance.cosine(data1_flat, data2_flat))
    # print(cosins[:20])

    print("cosine",np.mean(cosins))

    print("pearson_corr",np.mean(pearsons))

    # 6) 保存到 npz 文件 (rdm1, rdm2, rdm_correlation)
    # np.savez(args.output_result, rdm1=rdm1, rdm2=rdm2, rdm_correlation=corr)
    # print(f"Saved rdm1, rdm2, rdm_correlation={corr:.4f} to {args.output_result}")

if __name__ == "__main__":
    main()
