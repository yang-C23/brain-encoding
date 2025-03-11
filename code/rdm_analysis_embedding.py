import argparse
import os
import torch
import numpy as np
import pandas as pd
from scipy import spatial
import numpy, rsatoolbox

from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Compute embeddings, PPL, and RDM correlation for two Llama2 models.")
    parser.add_argument("--txt_path", type=str, required=True,
                        help="Path to the input .txt file (one sentence per line).")
    parser.add_argument("--model1", type=str, required=True,
                        help="Hugging Face path/name to the first Llama2-based model.")
    parser.add_argument("--model2", type=str, required=True,
                        help="Hugging Face path/name to the second Llama2-based model.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional Hugging Face token (if needed for private models).")
    parser.add_argument("--gpu", type=str, default="0",
                        help="Which GPU to use (via CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length for each input text.")
    # parser.add_argument("--output_csv", type=str, default="results.csv",
    #                     help="Output CSV file name.")
    return parser.parse_args()

def check_zero_weights(model, model_name=""):
    """
    (可选功能) 检查模型某些层是否有大量权重被置零。
    """
    with torch.no_grad():
        zero_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            if 'layers.' in name:
                zero_count += (param == 0).sum().item()
                total_count += param.numel()
        ratio = zero_count / total_count if total_count > 0 else 0.0
        print(f"[INFO] Ratio of zeroed weights in '{model_name} layers.*': {ratio:.4f}")
        if ratio < 0.0001:
            print(f"[WARNING] {model_name}: This model does not seem to be partially zeroed.")

def compute_embedding(model, tokenizer, text, device, max_length=128):
    """
    给定一个句子，返回该句子的 mean-pooled embedding。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 取最后一层 [batch_size, seq_len, hidden_dim]
        embedding = hidden_states.mean(dim=1).squeeze(0)  # mean pooling: [hidden_dim]
    return embedding.cpu().numpy()

def compute_perplexity(model, tokenizer, text, device, max_length=128):
    """
    计算给定文本在该模型下的 PPL (Perplexity)。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        # 这里使用 labels=inputs["input_ids"], 模型将自动计算交叉熵损失
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss  # CrossEntropy 平均损失
    ppl = torch.exp(loss)
    return ppl.item()

def compute_rdm(embeddings):
    """
    计算 NxN 的 RDM(表示差异矩阵)，使用相关距离：d = 1 - corr。
    embeddings: NumPy数组 [N, hidden_dim]
    """
    n = embeddings.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(embeddings[i], embeddings[j])[0, 1]
            rdm[i, j] = 1 - corr
    return rdm

def rdm_correlation(rdm1, rdm2):
    """
    计算两个 RDM 矩阵的相关系数（只取上三角部分）。
    """
    n = rdm1.shape[0]
    iu = np.triu_indices(n, k=1)
    rdm1_vals = rdm1[iu]
    rdm2_vals = rdm2[iu]
    return np.corrcoef(rdm1_vals, rdm2_vals)[0, 1]

def main():
    args = parse_args()

    # 指定要使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== 加载第一个模型 ========== 
    print(f"Loading Model1 from {args.model1}")
    tokenizer1 = AutoTokenizer.from_pretrained(
        args.model1,
        token=args.hf_token,
        trust_remote_code=True
    )
    # 对于Llama类模型，需要指定pad_token
    if tokenizer1.pad_token is None:
        tokenizer1.pad_token = tokenizer1.eos_token

    model1 = AutoModelForCausalLM.from_pretrained(
        args.model1,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True
    )
    check_zero_weights(model1, "Model1")
    model1.eval()

    # ========== 加载第二个模型 ========== 
    print(f"Loading Model2 from {args.model2}")
    tokenizer2 = AutoTokenizer.from_pretrained(
        args.model2,
        token=args.hf_token,
        trust_remote_code=True
    )
    if tokenizer2.pad_token is None:
        tokenizer2.pad_token = tokenizer2.eos_token

    model2 = AutoModelForCausalLM.from_pretrained(
        args.model2,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True
    )
    check_zero_weights(model2, "Model2")
    model2.eval()

    # ========== 读取TXT文件的全部文本 ========== 
    print(f"Reading txt from {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]  # 去掉空行

    # 用于存储输出到 CSV 的数据
    results = []

    # 收集 Embedding 用于后面计算 RDM
    embeddings_model1 = []
    embeddings_model2 = []

    # ========== 对每行文本计算PPL & Embedding ==========
    for text in lines:
        ppl1 = compute_perplexity(model1, tokenizer1, text, device, max_length=args.max_length)
        ppl2 = compute_perplexity(model2, tokenizer2, text, device, max_length=args.max_length)
        
        emb1 = compute_embedding(model1, tokenizer1, text, device, max_length=args.max_length)
        emb2 = compute_embedding(model2, tokenizer2, text, device, max_length=args.max_length)

        embeddings_model1.append(emb1)
        embeddings_model2.append(emb2)

        # 将该样本的信息存入 results（后续保存 CSV）
        results.append({
            "text": text,
            "ppl_model1": ppl1,
            "ppl_model2": ppl2
        })

    # ========== 转为 NumPy, 计算 RDM 和相关系数 ==========
    embeddings_model1 = np.array(embeddings_model1)  # [N, hidden_dim]
    embeddings_model2 = np.array(embeddings_model2)  # [N, hidden_dim]

    print("embed shape", embeddings_model2.shape)

    # d1 = rsatoolbox.data.Dataset(embeddings_model1)
    # d2 = rsatoolbox.data.Dataset(embeddings_model2)

    # rdms1 = rsatoolbox.rdm.calc_rdm(d1)
    # rdms2 = rsatoolbox.rdm.calc_rdm(d2)
    # corr = rsatoolbox.rdm.compare(rdms1,rdms2,method='corr')
    # print("corr", corr)



    N_filtered2 = embeddings_model2.shape[0]
    cosins = np.zeros(N_filtered2)
    print(cosins.shape)
    pearsons = np.zeros(N_filtered2)
    for i in range(N_filtered2):
        data1_flat = embeddings_model1[i].flatten()
        data2_flat = embeddings_model2[i].flatten()
        cos_sim = 1 - spatial.distance.cosine(data1_flat, data2_flat)
        cosins[i] = cos_sim
        pearson_corr = np.corrcoef(data1_flat, data2_flat)[0, 1]
        pearsons[i] = pearson_corr

    data1_flat = embeddings_model1[2].flatten()
    data2_flat = embeddings_model2[2].flatten()
    print(spatial.distance.cosine(data1_flat, data2_flat))
    print(cosins[:20])

    print("cosine",np.mean(cosins))

    print("pearson_corr",np.mean(pearsons))

    # if len(embeddings_model1) > 1:
    #     rdm1 = compute_rdm(embeddings_model1)
    #     rdm2 = compute_rdm(embeddings_model2)
    #     corr_rdm = rdm_correlation(rdm1, rdm2)
    # else:
    #     # 如果只有 1 个文本，那么 RDM 无法构建 NxN(只会是1x1)
    #     corr_rdm = np.nan

    # print("=== RESULTS ===")
    # print(f"RDM correlation between Model1 and Model2: {corr_rdm:.4f}")

    # # 将 RDM 相关系数也加入 CSV（可以作为最后一行，也可以单独存储）
    # results.append({
    #     "text": "RDM_correlation", 
    #     "ppl_model1": corr_rdm,   # 或者写到别的字段 
    #     "ppl_model2": ""
    # })

    # # ========== 保存到 CSV ==========
    # df_out = pd.DataFrame(results)
    # df_out.to_csv(args.output_csv, index=False, encoding='utf-8')
    # print(f"Saved CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
