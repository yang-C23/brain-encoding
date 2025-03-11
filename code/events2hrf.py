import os
import sys
import getopt
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

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

# 计算 frame_times
endpoint = nscans * tr
frame_times = np.arange(0.0, endpoint, tr)


ext = os.path.splitext(inputf)[1].lower()
if ext == ".npy":
    print(f"Loading NPY data from {inputf}")
    data_array = np.load(inputf)  # shape:(n_events, n_features)
    a = pd.DataFrame(data_array)
else:
    print(f"Loading CSV data from {inputf}")
    a = pd.read_csv(inputf, header=None)

n_events = len(a)
n_features = a.shape[1]

print(f"Data shape = {a.shape}, n_events={n_events}, n_features={n_features}")

# 第0列: onset
times = a.iloc[:, 0]

# 对每一列(1..end)做 HRF 卷积
first = True
for i in range(1, n_features):
    feature_i = a.iloc[:, i].values
    durations = np.zeros(n_events)

    x = compute_regressor(
        exp_condition=np.vstack((times, durations, feature_i)),
        hrf_model="spm",
        frame_times=frame_times,
        oversampling=2
    )

    if first:
        x1 = pd.DataFrame(x[0], columns=[f"col_{i}"])
        first = False
    else:
        x2 = pd.DataFrame(x[0], columns=[f"col_{i}"])
        x1 = x1.join(x2)

# 保存
x1.to_csv(outputf, header=None, index=False)
print(f"Saved HRF-convolved design matrix to {outputf} (shape={x1.shape})")