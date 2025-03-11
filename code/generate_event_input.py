import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def detect_embedding_column(df):
    """ 自动检测 embedding 列名 """
    for col in df.columns:
        if col.lower() in ["bert", "glove"]:
            return col
    raise ValueError("Embedding CSV 文件中没有找到 BERT 或 GloVe 列！")

def parse_embeddings(df, embedding_col):
    """ 解析嵌入向量字符串为 NumPy 数组 """
    embeddings_list = []
    print(f"Parsing {embedding_col} embeddings with progress bar...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=embedding_col):
        emb_str = row[embedding_col]  # 读取对应的 embedding 字符串
        emb_clean = emb_str.strip('"[]')  # 去掉引号和方括号
        vals = [float(x) for x in emb_clean.split()]  # 转换成浮点数列表
        embeddings_list.append(vals)
    return np.array(embeddings_list)

def main():
    parser = argparse.ArgumentParser(description="Process word embeddings and save as structured format.")
    parser.add_argument("-embedding", type=str, required=True, help="Path to the embedding CSV file (BERT/GloVe)")
    parser.add_argument("-information", type=str, required=True, help="Path to the word information CSV file")
    parser.add_argument("-output", type=str, required=True, help="Output file path (.npy or .csv)")
    
    args = parser.parse_args()
    
    # 1) 读入词信息表
    df_info = pd.read_csv(args.information)
    
    # 2) 读入 embedding 文件
    df_emb = pd.read_csv(args.embedding)
    
    # 3) 识别 embedding 列名
    embedding_col = detect_embedding_column(df_emb)
    
    # 4) 合并数据
    df_merged = pd.merge(df_info, df_emb, on="word", how="inner")
    
    # 5) 解析嵌入向量
    embeddings = parse_embeddings(df_merged, embedding_col)
    
    # 6) 生成最终数据 (onset + embeddings)
    data_for_events = np.column_stack([df_merged["onset"].values, embeddings])
    print(f"Final array shape = {data_for_events.shape}")
    
    # 7) 根据输出文件格式保存数据
    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext == ".npy":
        np.save(args.output, data_for_events)
        print(f"Saved to {args.output} (NumPy .npy format).")
    elif output_ext == ".csv":
        pd.DataFrame(data_for_events).to_csv(args.output, header=None, index=False, float_format="%.6f")
        print(f"Saved to {args.output} (CSV format).")
    else:
        raise ValueError("Unsupported output format. Use .npy or .csv")

if __name__ == "__main__":
    main()
