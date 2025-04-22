#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Generate word embeddings with context window from Llama2 model.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path or name of the Llama2 model for tokenizer, e.g. 'meta-llama/Llama-2-7b-hf'")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the word information CSV file, must have columns [word, onset, offset]")
    parser.add_argument("--output", type=str, default="events_word_context.npy",
                        help="Output .npy file name (default: events_word_context.npy)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"),
                        help="Hugging Face access token (default: use HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "fr", "zh"],
                        help="Language of the input words (en: English, fr: French, zh: Chinese)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID(s) to use (default: 0). Set to '0,1' for multi-GPU.")
    parser.add_argument("--context-size", type=int, default=100,
                        help="Number of words to use as context before the target word (default: 100)")
    return parser.parse_args()


def build_context_window(df, target_idx, context_size=100):
    """Build context window for target word at target_idx."""
    # Get all words up to and including the target word
    end_idx = min(target_idx + 1, len(df))
    start_idx = max(0, end_idx - context_size)
    
    # Get context words and convert to string, replacing NaN with empty string
    context_words = df['word'].iloc[start_idx:end_idx].fillna('').astype(str).tolist()
    context_text = ' '.join(context_words)
    
    # Record the position of the target word in the joined text
    target_word = str(df['word'].iloc[target_idx])
    if pd.isna(df['word'].iloc[target_idx]):
        target_word = ""
    
    return context_text, target_word


def get_token_indices(tokenizer, text, target_word):
    """Get indices of tokens that represent the target word."""
    # Tokenize the whole text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_words = tokenizer.convert_ids_to_tokens(tokens)
    
    # Find the target word tokens at the end of the sequence
    target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    target_token_words = tokenizer.convert_ids_to_tokens(target_tokens)
    
    # Check if the target word is at the end by matching tokens
    matches = []
    for i in range(len(token_words) - len(target_token_words) + 1):
        match = True
        for j in range(len(target_token_words)):
            if i + j >= len(token_words) or token_words[i + j] != target_token_words[j]:
                match = False
                break
        if match:
            matches.append((i, i + len(target_token_words)))
    
    # Return the last match (should be our target word at the end)
    if matches:
        return matches[-1]
    
    # Fallback: return the last token(s)
    print(f"Warning: Could not identify '{target_word}' tokens. Using last token as fallback.")
    return (len(tokens) - 1, len(tokens))


def main():
    args = parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer first
    print(f"Loading tokenizer...")
    #tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=args.hf_token, trust_remote_code=True)
   
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully!")
    print(f"Starting to load model from {args.model}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True
    )

    # Check if model might be quantized/partially zeroed
    with torch.no_grad():
        zero_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            if 'layers.' in name:
                zero_count += (param == 0).sum().item()
                total_count += param.numel()
        ratio = zero_count / total_count
        print(f"[INFO] Ratio of zeroed weights in 'layers.*': {ratio:.4f}")
        if ratio < 0.0001:
            print("[WARNING] This model does not seem to be partially zeroed.")

    # Switch to evaluation mode
    model.eval()

    # Read CSV
    print(f"Reading CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    required_cols = ['word', 'onset', 'offset']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    # Generate embeddings for each word
    embeddings_list = []
    onsets_list = []
    durations_list = []

    for idx in range(len(df)):

        #word = df['word'].iloc[idx] 因为word会有空格，所以用lemma. e.g haven't -> have not
        #word = df['word'].iloc[idx]

        # Handle NaN words
        if pd.isna(df['word'].iloc[idx]):
            print(f"Skipping NaN word at index {idx+1}/{len(df)}")
            continue
            
        word = str(df['word'].iloc[idx])
        
        # Check if onset/offset are valid
        if pd.isna(df['onset'].iloc[idx]) or pd.isna(df['offset'].iloc[idx]):
            print(f"Skipping word '{word}' with invalid onset/offset at index {idx+1}/{len(df)}")
            continue
        # Handle NaN words

        
        onset = float(df['onset'].iloc[idx])
        offset = float(df['offset'].iloc[idx])
        duration = offset - onset
        
        #print(f"Processing word: {word} (index {idx+1}/{len(df)})")
        
        # Build context window
        context_text, target_word = build_context_window(df, idx, args.context_size)
        
        # Skip if target word is empty
        if not target_word:
            print(f"Skipping empty target word at index {idx+1}/{len(df)}")
            continue
            
        # Get token indices for the target word
        start_token_idx, end_token_idx = get_token_indices(tokenizer, context_text, target_word)
        
        # Tokenize the context
        inputs = tokenizer(
            context_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Adjust based on model's max context
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]  # Last layer, first batch
        
        # Extract and mean-pool the target word's token embeddings
        word_embedding = hidden_states[start_token_idx:end_token_idx].mean(dim=0).cpu().numpy()
        
        embeddings_list.append(word_embedding)
        onsets_list.append(onset)
        durations_list.append(duration)

    # Check if we have any valid words
    if not embeddings_list:
        print("Error: No valid words were processed. Check your input data.")
        sys.exit(1)

    # Stack embeddings, onsets, and durations
    embeddings_arr = np.vstack(embeddings_list)
    onsets_arr = np.array(onsets_list)[:, None]
    durations_arr = np.array(durations_list)[:, None]
    
    # Format: [onset, duration, embedding_dim1, embedding_dim2, ...]
    events_array = np.hstack([onsets_arr, durations_arr, embeddings_arr])
    
    # Save to NPY
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(args.output, events_array)
    print(f"Saved events array with shape={events_array.shape} to {args.output}")
    print(f"Format: [onset, duration, embedding_features...]")


if __name__ == "__main__":
    main() 