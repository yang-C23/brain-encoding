from transformers import AutoConfig, LlamaForCausalLM
import torch

# 1. 加载 Llama-2-7B 的配置
#    这里会从 "meta-llama/Llama-2-7b-hf" 的 config.json 中读取模型结构相关信息
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 使用该配置创建一个随机初始化的 LlamaForCausalLM 模型
model = LlamaForCausalLM(config)

# 3. 将随机初始化的模型保存为 Hugging Face 格式
#    这样就能像普通权重文件一样使用 save_pretrained / from_pretrained
model.save_pretrained("brain_encoding/model/llama-2-7b-random-initialization")

print("模型已保存到 'brain_encoding/model/llama-2-7b-random-initialization'")
