# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate events npy from Llama2-7B embeddings.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path or name of the Llama2-7B model, e.g. 'meta-llama/Llama-2-7b-hf' or local dir")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the information CSV file, which must have columns [lemma, onset]")
    parser.add_argument("--output", type=str, default="events_llama2.npy",
                        help="Output .npy file name (default: events_llama2.npy)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"),
                        help="Hugging Face access token (default: use HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "fr", "zh"],
                        help="Language of the input lemmas (en: English, fr: French, zh: Chinese)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID(s) to use (default: 0). Set to '0,1' for multi-GPU.")
    return parser.parse_args()


def preprocess_lemma(lemma: str, lang: str) -> str:
    """Preprocess lemma based on language."""
    if lang == "zh":
        return lemma  
    elif lang == "fr":
        return lemma  
    else:
        return lemma 


def main():
    args = parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Llama2 model and tokenizer
    print(f"Loading Llama2 model from {args.model} ...")
    # 修改后的代码片段（在加载分词器后添加两行）
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # 添加以下两行代码
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用结束符作为填充符
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True
    )#.to(device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     trust_remote_code=True
    # )#.to(device)


    with torch.no_grad():
        zero_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            # 可以只检查某些关键层，比如 decoder.layers.xx.* ，也可以全部都检查
            if 'layers.' in name:
                zero_count += (param == 0).sum().item()
                total_count += param.numel()
        ratio = zero_count / total_count
        print(f"[INFO] Ratio of zeroed weights in 'layers.*': {ratio:.4f}")
        if ratio < 0.0001:  # 如果几乎没有被置零
            print("[WARNING] This model does not seem to be partially zeroed.")


    # Switch to evaluation mode
    model.eval()

    # Read CSV
    print(f"Reading CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    if "sentence" not in df.columns or "onset" not in df.columns  or "duration" not in df.columns:
        raise ValueError("CSV must contain 'sentence', 'onset' and 'duration' columns!")

    # Generate embeddings for each lemma
    embeddings_list = []
    onsets_list = []
    duration_list = []

    for idx, row in df.iterrows():
        sentence_str = str(row["sentence"])
        onset_val = float(row["onset"])
        duration_val = float(row["duration"])

        # Preprocess lemma based on language
        processed_sentence = preprocess_lemma(sentence_str, args.lang)

        # Tokenize
        inputs = tokenizer(
            processed_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Single word doesn't need long sequence
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last hidden layer

        # Use mean pooling or first token embedding
        emb = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()  # Mean pooling
        # emb = hidden_states[:, 0, :].squeeze(0).cpu().numpy()  # First token embedding

        embeddings_list.append(emb)
        onsets_list.append(onset_val)
        duration_list.append(duration_val)

    # Stack embeddings and onsets
    embeddings_arr = np.vstack(embeddings_list)  # (N, hidden_dim)
    onsets_arr = np.array(onsets_list)[:, None]  # (N, 1)
    duration_arr = np.array(duration_list)[:,None]

    print("embeddings_arr shape", embeddings_arr.shape)
    print("onsets_arr shape", onsets_arr.shape)
    print("duration_arr shape", duration_arr.shape)

    events_array = np.hstack([onsets_arr,duration_arr, embeddings_arr])  # (N, 1+1+hidden_dim)
    print("events_array shape", events_array.shape)
    # Save to NPY
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output, events_array)
    print(f"Saved events array shape={events_array.shape} to {args.output}")


if __name__ == "__main__":
    main()