
import os
import re
import nibabel as nib

def find_bold_runs(func_dir):
    """
    在给定的 func_dir 下，查找所有符合
    'sub-XXX_task-???_run-??_space-MNIColin27_desc-preproc_bold.nii.gz'
    形式的文件，并返回 (run_number, filepath) 列表。
    run_number 根据文件名里的 'run-XX' 解析。
    """
    pattern = re.compile(r'run-(\d+)')
    runs_found = []

    for fname in os.listdir(func_dir):
        # 只考虑 .nii.gz 文件，且包含 space-MNIColin27_desc-preproc_bold
        if fname.endswith('.nii.gz') and 'space-MNIColin27_desc-preproc_bold' in fname:
            match = pattern.search(fname)
            if match:
                run_num_str = match.group(1)  # run id
                run_num = int(run_num_str)
                full_path = os.path.join(func_dir, fname)
                runs_found.append((run_num, full_path))

    # 按 run 号升序排序
    runs_found.sort(key=lambda x: x[0])
    return runs_found


def concat_runs_for_sub(sub_id, derivatives_dir="derivatives", out_dir="derivatives-merged"):
    """
    对指定被试 sub_id，在 derivatives/sub_id/func/ 下找到所有 run 文件，
    按顺序拼接，并输出到 out_dir/sub_id/func/ 中。
    文件名里包含总时间点数。
    """
    func_path = os.path.join(derivatives_dir, sub_id, "func")
    if not os.path.isdir(func_path):
        print(f"[Skipping] {func_path} not found or is not a directory.")
        return

    runs = find_bold_runs(func_path)
    if len(runs) == 0:
        print(f"[Skipping] No BOLD runs found in {func_path}.")
        return

    # 读取并拼接
    images_to_concat = []
    total_timepoints = 0
    for run_num, fpath in runs:
        print(f"  Loading run-{run_num:02d}: {fpath}")
        img = nib.load(fpath)
        shape_4d = img.shape
        if len(shape_4d) != 4:
            print(f"    [Error] File is not 4D: shape={shape_4d}. Skip.")
            continue

        tpoints = shape_4d[-1]
        total_timepoints += tpoints
        images_to_concat.append(img)

    if len(images_to_concat) == 0:
        print(f"[Skipping] All runs in {func_path} were invalid.")
        return

    # 拼接
    print(f"  Concatenating {len(images_to_concat)} runs along time dimension...")
    merged_img = nib.concat_images(images_to_concat, axis=3)
    merged_shape = merged_img.shape

    # 准备输出目录
    out_sub_func_dir = os.path.join(out_dir, sub_id, "func")
    os.makedirs(out_sub_func_dir, exist_ok=True)

    # 组装输出文件名，让它包含 total_timepoints
    # 这里简单用 sub-XXX_task-lpp_mergedTP-<timepoints>.nii.gz 格式，
    # 可根据需要再自行定制
    out_fname = f"{sub_id}_task-lppCN_mergedTP-{total_timepoints}_bold.nii.gz"
    out_fpath = os.path.join(out_sub_func_dir, out_fname)

    # 保存结果
    nib.save(merged_img, out_fpath)
    print(f"  => Saved merged file: {out_fpath}")
    print(f"  => Merged shape = {merged_shape}")
    print(f"  => Final total timepoints = {total_timepoints}\n")


def main():
    derivatives_dir = "brain_encoding/ds003643-download/derivatives"
    out_dir = "derivatives-merged"

    # 扫描 derivatives_dir 下所有子文件夹 sub-XXX
    # 只要名字形如 'sub-XXX' 就处理
    # 你也可以根据实际需要过滤比如 sub-CNXXX, sub-ENXXX etc.
    for entry in os.listdir(derivatives_dir):
        sub_path = os.path.join(derivatives_dir, entry)
        if os.path.isdir(sub_path) and entry.startswith("sub-"):
            print(f"\nProcessing subject: {entry}")
            concat_runs_for_sub(entry, derivatives_dir, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
