
import os
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    hf_token = os.environ.get('hf_token')
    if not hf_token:
        raise ValueError("hf_token not found in .env file")
    login(token=hf_token)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    return model, tokenizer, device