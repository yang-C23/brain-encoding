#输入：假定有两个CSV文件：

# sentences.csv：包含句子信息，每行包含字段 sentence_id、onset（句子开始时间，秒）和 duration（句子持续时间，秒）。
# sentence_vectors.csv：每行对应一个句子的向量表示，行索引或第一列对应 sentence_id（可根据需要调整）。

#总体思路：
# 1 构建高分辨率时间轴
# 根据 fMRI 的采集参数（TR 和扫描次数），在比 TR 更细的时间尺度上构造连续的高分辨率时间轴，以便精细捕捉句子刺激在时间上的分布。

# 2 生成刺激信号
# 从句子信息（包括句子的起始时间和持续时间）以及对应的句向量出发，在高分辨率时间轴上为每个句子标记其出现的时间区间。如果多个句子在同一时间点重叠，利用简单平均或加权策略合成该时间点的句向量表示。

# 3 HRF 卷积
# 使用一个 HRF 模型（例如 SPM 的 HRF，通过 spm_hrf 生成）对高分辨率的刺激信号进行卷积。卷积过程相当于将刺激信号通过生理模型进行平滑，模拟神经活动到血流动力学响应之间的转换。

# 4 降采样对齐 fMRI
# 卷积后得到的是高分辨率的连续信号，通过降采样（根据 oversampling 因子）将卷积结果映射到实际 fMRI 采集时间点上，得到与 fMRI 数据对齐的设计矩阵。
import os
import numpy as np
import pandas as pd
from nilearn.glm.first_level import spm_hrf
import matplotlib.pyplot as plt
import sys
import getopt


def usage():
    print("Generate predicted hrf (and optional derivative) from an events file")
    print("""
    Usage:
      events2hrf.py -t <TR> -n <nscans> -i <input> -o <output>

    -i input_file : either a CSV or NPY file with columns [onset, amplitude1, amplitude2, ...]
    -t xx         : xx is the TR in seconds
    -n nn         : nn is the total number of scans
    -o output_file: CSV file containing the generated timecourse(s)

Example:
    events2hrf.py -t 2.0 -n 600 -i f0_short.csv -o f0_reg.csv
    events2hrf.py -t 2.0 -n 2977 -i glove_events_input.npy -o glove_regressors.csv
""")

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:n:hi:o:", 
                               ["tr=", "nscans=", "help", "input=", "output="])
except getopt.GetoptError as err:
    print(str(err))  
    usage()
    sys.exit(2)

# -------------------------------
# 参数设置
# -------------------------------
# TR = 2.0               # fMRI的重复时间，单位秒
# nscans = 600           # 扫描次数
# endpoint = nscans * TR # fMRI采集总时长

outputf = "output.csv"
inputf = None
tr = None
nscans = None

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

if inputf is None or tr is None or nscans is None:
    usage()
    sys.exit(2)

# -------------------------------
# 加载句子信息与句向量
# -------------------------------
# sentences.npy包含：onset, duration, sentence embedding

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
# 参数设置
# -------------------------------

# oversampling因子：高分辨率时间步数，每个TR划分为若干小步
oversampling_factor = 2  
dt_highres = tr / oversampling_factor  # 高分辨率时间步长
endpoint = nscans * tr

# 用户选项：选择方法，"lag_matrix" 或 "simple_shift"
method = "lag_matrix"  # 可选 "lag_matrix" 或 "simple_shift"

# 对于简单延后法，设置延后的 TR 数（例如 2 个 TR）
fixed_delay_TR = 2

# 对于设计矩阵法，设置最大滞后数（例如 10 个 TR）
max_lag = 4
print("max_lag",max_lag)


# -------------------------------
# 构造高分辨率刺激信号矩阵
# -------------------------------
# 构建高分辨率时间轴（单位：秒）
highres_time = np.arange(0.0, endpoint, dt_highres)
n_highres = len(highres_time)

# 初始化高分辨率刺激矩阵，形状为 (n_highres, vector_dim)
stimulus_highres = np.zeros((n_highres, vector_dim), dtype=float)
# 记录每个时间点累积的权重，用于处理重叠（后续做平均）
weights = np.zeros(n_highres, dtype=float)

# 对每个句子，根据其onset和duration填充刺激信号
for i in range(len(onsets)):
    onset = onsets[i]
    duration = durations[i]
    emb_vec = embeddings[i]
    


    # 找出高分辨率时间轴上该句子覆盖的索引
    # 注意：如果句子结束时间超过了采集总时长，则只取有效部分
    mask = (highres_time >= onset) & (highres_time < min(onset + duration, endpoint))
    
    # 将句向量累加到对应的时间点上
    stimulus_highres[mask, :] += emb_vec
    # 同时累加权重，方便后续求平均（若存在重叠）
    weights[mask] += 1.0

# 对每个时间点，若有重叠则求平均
# 注意防止除以0
nonzero_mask = weights > 0
stimulus_highres[nonzero_mask, :] /= weights[nonzero_mask][:, np.newaxis]


# -------------------------------
# 下采样到 fMRI 的 TR 分辨率
# -------------------------------
# 简单取每 oversampling_factor 个点
stimulus_TR = stimulus_highres[::oversampling_factor, :]

# 如果下采样后的数据点数与 nscans 不符，则做截断或边缘填充（用最后一行填充）
n_current = stimulus_TR.shape[0]
if n_current > nscans:
    stimulus_TR = stimulus_TR[:nscans, :]
elif n_current < nscans:
    missing = nscans - n_current
    # 使用最后一行作为填充值
    last_row = stimulus_TR[-1, :]
    pad_rows = np.tile(last_row, (missing, 1))
    stimulus_TR = np.vstack([stimulus_TR, pad_rows])

print("Downsampled stimulus shape (TR resolution):", stimulus_TR.shape)


# -------------------------------
# 构建最终的设计矩阵或延后信号
# -------------------------------
if method == "lag_matrix":
    # 构建时间滞后设计矩阵
    # 设计矩阵 X 的尺寸为 (nscans, vector_dim * max_lag)
    X = np.zeros((nscans, vector_dim * max_lag))
    for lag in range(max_lag):
        # 对于滞后 lag，刺激信号向下平移 lag 个 TR
        # 对于前 lag 个时间点，无法获得完整信息，默认填0（先填0试一下吧）
        if lag == 0:
            X[:, lag*vector_dim:(lag+1)*vector_dim] = stimulus_TR
        else:
            X[lag:, lag*vector_dim:(lag+1)*vector_dim] = stimulus_TR[:-lag, :]
    design_matrix = X
    print("Constructed lagged design matrix shape:", design_matrix.shape)
    
elif method == "simple_shift":
    # 直接将刺激信号整体延后固定的 TR 数
    # 对于前 fixed_delay_TR 个 TR 没有对应信息，用0填充（先填0试一下吧）
    design_matrix = np.zeros_like(stimulus_TR)
    if fixed_delay_TR < nscans:
        design_matrix[fixed_delay_TR:, :] = stimulus_TR[:-fixed_delay_TR, :]
    else:
        print("Warning: fixed_delay_TR is larger than total number of scans!")
    print("Constructed simple shifted design matrix shape:", design_matrix.shape)
else:
    raise ValueError("method 参数必须为 'lag_matrix' 或 'simple_shift'")



# -------------------------------
# 保存结果到CSV文件
# -------------------------------
# output_filename = "sentence_hrf_design.csv"
# 将设计矩阵转换为DataFrame保存
output_dir = os.path.dirname(outputf)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df_design = pd.DataFrame(design_matrix)
df_design.to_csv(outputf, index=False, header=False)
print(f"Saved HRF-convolved design matrix to {outputf} with shape {design_matrix.shape}")

# -------------------------------
# 可选：绘制部分结果以便检验
# -------------------------------
# image_dir = os.path.join(os.path.dirname(outputf), "image")
# os.makedirs(image_dir, exist_ok=True)
# plot_path = os.path.join(image_dir, "hrf_preview2.png")

# plt.figure(figsize=(10, 4))
# plt.plot(np.arange(nscans), design_matrix[:, 0], label="Dimension 0")
# plt.xlabel("TR index")
# plt.ylabel("Convolved amplitude")
# plt.title("HRF-convolved signal (Dimension 0)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(plot_path, dpi=150)
# plt.close()  # 关闭画布，不再弹出窗口
