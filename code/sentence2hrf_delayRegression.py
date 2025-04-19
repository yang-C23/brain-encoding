# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
#from nilearn.glm.first_level import spm_hrf
#import matplotlib.pyplot as plt
import sys
import getopt

def usage():
    print("Generate predicted hrf (and optional derivative) from an events file")
    print("""
    Usage:
      events2hrf.py -t <TR> -n <nscans> -i <input> -o <output> [--method <lag_matrix|simple_shift>] [--lag <int>]

    -i, --input <file>        : NPY file with columns [onset, duration, embedding...]
    -t, --tr <float>          : TR in seconds
    -n, --nscans <int>        : total number of scans
    -o, --output <file>       : CSV file containing the generated timecourse(s)
    --method <str>            : "lag_matrix" (default) or "simple_shift"
    --lag <int>               : 
                                if method=lag_matrix -> this is max_lag
                                if method=simple_shift -> this is the TR delay

Example:
    events2hrf.py -t 2.0 -n 300 -i example.npy -o output.csv
    events2hrf.py -t 2.0 -n 300 -i example.npy -o output.csv --method lag_matrix --lag 6
    events2hrf.py -t 2.0 -n 300 -i example.npy -o output.csv --method simple_shift --lag 3
""")

try:
    # 新增 --method, --lag
    opts, args = getopt.getopt(
        sys.argv[1:], 
        "t:n:hi:o:", 
        ["tr=", "nscans=", "help", "input=", "output=",
         "method=", "lag="]
    )
except getopt.GetoptError as err:
    print(str(err))  
    usage()
    sys.exit(2)

# -------------------------------
# 参数初始化
# -------------------------------
outputf = "output.csv"
inputf = None
tr = None
nscans = None

method = "lag_matrix"   # 默认方法
lag_value = None        # 由命令行指定，为 None 时使用默认值

# lag_matrix 默认值、simple_shift 默认值
default_max_lag = 4
default_delay_tr = 2

for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit(2)
    elif o in ("-t", "--tr"):
        tr = float(a) 
    elif o in ("-n", "--nscans"):
        nscans = int(a)
    elif o in ("-o", "--output"):
        outputf = a
    elif o in ("-i", "--input"):
        inputf = a
    elif o == "--method":
        if a not in ["lag_matrix", "simple_shift"]:
            print(f"[ERROR] Invalid method: {a}")
            usage()
            sys.exit(2)
        method = a
    elif o == "--lag":
        lag_value = int(a)

if inputf is None or tr is None or nscans is None:
    usage()
    sys.exit(2)

# 根据 method 设置 max_lag 或 fixed_delay_TR
if method == "lag_matrix":
    if lag_value is None:
        max_lag = default_max_lag
    else:
        max_lag = lag_value
    print(f"[INFO] method = lag_matrix, max_lag = {max_lag}")

elif method == "simple_shift":
    if lag_value is None:
        fixed_delay_TR = default_delay_tr
    else:
        fixed_delay_TR = lag_value
    print(f"[INFO] method = simple_shift, delay_tr = {fixed_delay_TR}")

# -------------------------------
# 加载句子信息与句向量
# -------------------------------
ext = os.path.splitext(inputf)[1].lower()
if ext == ".npy":
    print(f"[INFO] Loading NPY data from {inputf}")
    data_array = np.load(inputf)  # shape: (n_events, 2 + embedding_dim)
    # 分拆: 前两列为 onset & duration，余下列为句向量
    onsets = data_array[:, 0]
    durations = data_array[:, 1]
    embeddings = data_array[:, 2:]
    vector_dim = embeddings.shape[1]
    print(f"[INFO] Loaded {len(onsets)} events, each embedding dim={vector_dim}")
else:
    print(f"[ERROR] Only NPY supported in this script. Got: {ext}")
    usage()
    sys.exit(2)

# -------------------------------
# 相关参数设置
# -------------------------------
endpoint = nscans * tr
oversampling_factor = 2  
dt_highres = tr / oversampling_factor  # 高分辨率时间步长

# 构建高分辨率时间轴（单位：秒）
highres_time = np.arange(0.0, endpoint, dt_highres)
n_highres = len(highres_time)

# -------------------------------
# 构建高分辨率刺激信号矩阵
# -------------------------------
stimulus_highres = np.zeros((n_highres, vector_dim), dtype=float)
weights = np.zeros(n_highres, dtype=float)

for i in range(len(onsets)):
    onset = onsets[i]
    duration = durations[i]
    emb_vec = embeddings[i]
    
    mask = (highres_time >= onset) & (highres_time < min(onset + duration, endpoint))
    
    stimulus_highres[mask, :] += emb_vec
    weights[mask] += 1.0

nonzero_mask = weights > 0
stimulus_highres[nonzero_mask, :] /= weights[nonzero_mask][:, np.newaxis]

# -------------------------------
# 下采样到 fMRI 的 TR 分辨率
# -------------------------------
stimulus_TR = stimulus_highres[::oversampling_factor, :]
n_current = stimulus_TR.shape[0]
if n_current > nscans:
    stimulus_TR = stimulus_TR[:nscans, :]
elif n_current < nscans:
    missing = nscans - n_current
    last_row = stimulus_TR[-1, :]
    pad_rows = np.tile(last_row, (missing, 1))
    stimulus_TR = np.vstack([stimulus_TR, pad_rows])

print(f"[INFO] Downsampled stimulus shape (TR resolution): {stimulus_TR.shape}")

# -------------------------------
# 根据 method 构建最终的设计矩阵
# -------------------------------
if method == "lag_matrix":
    # 构建时间滞后设计矩阵
    design_matrix = np.zeros((nscans, vector_dim * max_lag))
    for lag in range(max_lag):
        if lag == 0:
            design_matrix[:, lag*vector_dim:(lag+1)*vector_dim] = stimulus_TR
        else:
            design_matrix[lag:, lag*vector_dim:(lag+1)*vector_dim] = stimulus_TR[:-lag, :]
    print(f"[INFO] Constructed lagged design matrix shape: {design_matrix.shape}")

elif method == "simple_shift":
    # 整体延后 fixed_delay_TR 个 TR
    design_matrix = np.zeros_like(stimulus_TR)
    if fixed_delay_TR < nscans:
        design_matrix[fixed_delay_TR:, :] = stimulus_TR[:-fixed_delay_TR, :]
    else:
        print("[WARN] fixed_delay_TR >= total number of scans, entire design_matrix is zero!")
    print(f"[INFO] Constructed simple shifted design matrix shape: {design_matrix.shape}")

# -------------------------------
# 保存结果到 CSV
# -------------------------------
output_dir = os.path.dirname(outputf)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df_design = pd.DataFrame(design_matrix)
df_design.to_csv(outputf, index=False, header=False)
print(f"[INFO] Saved design matrix to {outputf} with shape {design_matrix.shape}")

# -------------------------------
# 可选可视化
# -------------------------------
# image_dir = os.path.join(os.path.dirname(outputf), "image")
# os.makedirs(image_dir, exist_ok=True)
# plot_path = os.path.join(image_dir, "hrf_preview.png")
# plt.figure(figsize=(10, 4))
# plt.plot(np.arange(nscans), design_matrix[:, 0], label="Dimension 0")
# plt.xlabel("TR index")
# plt.ylabel("Convolved amplitude")
# plt.title("HRF-convolved signal (Dimension 0)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(plot_path, dpi=150)
# plt.close()
