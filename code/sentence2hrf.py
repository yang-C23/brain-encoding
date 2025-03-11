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
# 构造高分辨率刺激信号矩阵
# -------------------------------

# oversampling因子：高分辨率时间步数，每个TR划分为若干小步
oversampling_factor = 2  
dt_highres = tr / oversampling_factor  # 高分辨率时间步长
endpoint = nscans * tr

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
# HRF卷积
# -------------------------------
# 利用spm_hrf生成HRF曲线（在高分辨率下计算）
# spm_hrf的输入为时间步长 dt_highres
hrf_curve = spm_hrf(dt_highres)

# 为每个句向量的维度进行卷积
# 初始化最终的设计矩阵：每个fMRI采集点对应一个向量
hrf_design = np.zeros((nscans, vector_dim), dtype=float)

# 对高分辨率刺激信号每个维度做卷积，再降采样到TR时间点
for d in range(vector_dim):
    # 取出高分辨率下某一维度的信号
    signal_highres = stimulus_highres[:, d]
    
    # 使用numpy.convolve进行卷积，'full'模式返回长度为 n_highres + len(hrf_curve) - 1
    # convolved_full = np.convolve(signal_highres, hrf_curve, mode='full') * dt_highres
    # 截取与高分辨率时间轴长度相同的部分
    # convolved_highres = convolved_full[:n_highres]
    
    # 降采样：取每 oversampling_factor 个点作为一个fMRI采集点
    # convolved_tr = convolved_highres[::oversampling_factor]
    
    # 确保降采样后长度与nscans匹配（可能存在小数点误差）
    # if len(convolved_tr) > nscans:
    #     convolved_tr = convolved_tr[:nscans]
    # elif len(convolved_tr) < nscans:
    #     # 若不足则补0
    #     convolved_tr = np.pad(convolved_tr, (0, nscans - len(convolved_tr)), mode='constant')
    
    # 存储到设计矩阵的对应维度
    # hrf_design[:, d] = convolved_tr


    # -------------------------------
    #不使用hrf卷积，直接用刺激信号做延迟然后输出
    # -------------------------------
    shift_scans = 2

    shift_size = shift_scans * oversampling_factor
    signal_shifted = np.pad(signal_highres, (shift_size, 0), mode='constant')[:n_highres]

    signal_tr = signal_shifted[::oversampling_factor]
    if len(signal_tr) > nscans:
        signal_tr = signal_tr[:nscans]
    elif len(signal_tr) < nscans:
        signal_tr = np.pad(signal_tr, (0, nscans - len(signal_tr)), mode='constant')
    
    hrf_design[:, d] = signal_tr





# -------------------------------
# 保存结果到CSV文件
# -------------------------------
# output_filename = "sentence_hrf_design.csv"
# 将设计矩阵转换为DataFrame保存
output_dir = os.path.dirname(outputf)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df_design = pd.DataFrame(hrf_design)
df_design.to_csv(outputf, index=False, header=False)
print(f"Saved HRF-convolved design matrix to {outputf} with shape {hrf_design.shape}")

# -------------------------------
# 可选：绘制部分结果以便检验
# -------------------------------
image_dir = os.path.join(os.path.dirname(outputf), "image")
os.makedirs(image_dir, exist_ok=True)
plot_path = os.path.join(image_dir, "hrf_preview2.png")

plt.figure(figsize=(10, 4))
plt.plot(np.arange(nscans), hrf_design[:, 0], label="Dimension 0")
plt.xlabel("TR index")
plt.ylabel("Convolved amplitude")
plt.title("HRF-convolved signal (Dimension 0)")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()  # 关闭画布，不再弹出窗口
