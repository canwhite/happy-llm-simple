import os
import torch
from contextlib import nullcontext
from k_model import ModelConfig, Transformer
from transformers import AutoTokenizer


class TextGenerator:
    def __init__(self, 
                 checkpoint_path='./output/pretrain_dim1024_layers18_vocab_size6144.pth',
                 tokenizer_path='./tokenizer_k/',
                 seed=42,
                 device=None,
                 dtype="bfloat16"):
        
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        self.model_config = ModelConfig(dim=1024, n_layers=18, vocab_size=6144)
        self.model = Transformer(self.model_config)
        
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
            unwanted_prefix = '_orig_mod.'
            for k, v in list(checkpoint_dict.items()):
                if k.startswith(unwanted_prefix):
                    checkpoint_dict[k[len(unwanted_prefix):]] = checkpoint_dict.pop(k)
            self.model.load_state_dict(checkpoint_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint file {checkpoint_path} not found. Using uninitialized model.")
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手，你的名字叫小明。"},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def sft_sample(self, 
                   start="Hello!", 
                   num_samples=3, 
                   max_new_tokens=256, 
                   temperature=0.7, 
                   top_k=300):
        
        start = self.chat_template(start)
        start_ids = self.tokenizer(start, return_tensors="pt").input_ids
        x = start_ids.to(self.device)
        
        generated_texts = []
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, stop_id=self.tokenizer.eos_token_id, 
                                           max_new_tokens=max_new_tokens, 
                                           temperature=temperature, 
                                           top_k=top_k)
                    generated_texts.append(self.tokenizer.decode(y[0], skip_special_tokens=True))
        return generated_texts

    def pretrain_sample(self, 
                        start="Hello!", 
                        num_samples=3, 
                        max_new_tokens=256, 
                        temperature=0.7, 
                        top_k=300):
        
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        
        start_ids = self.tokenizer(start, return_tensors="pt").input_ids
        x = start_ids.to(self.device)
        
        generated_texts = []
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, 
                                           temperature=temperature, 
                                           top_k=top_k)
                    generated_texts.append(self.tokenizer.decode(y[0], skip_special_tokens=True))
        
        return generated_texts


if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")

    pretrain_prompt_datas = [
        '北京大学是',
        '中国矿业大学（北京）地球科学与测绘工程学院',
    ]

    generator = TextGenerator(checkpoint_path='./output/pretrain_dim1024_layers18_vocab_size6144.pth')
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, 
                                           max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i+1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-'*20}")

    print("\n ------------------- SFT Sample ------------------- \n")

    sft_prompt_datas = [
        '你好呀',
        "中国的首都是哪里？",
        "1+1等于多少？",
        "你是谁？"
    ]
    
    generator = TextGenerator(checkpoint_path='./output/sft_dim1024_layers18_vocab_size6144.pth')
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1, 
                                      max_new_tokens=128, temperature=0.6)
        print(f"\nSample {i+1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-'*20}")