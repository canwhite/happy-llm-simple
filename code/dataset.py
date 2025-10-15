import os
import random
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token if self.tokenizer.bos_token else ''}{sample['text']}"
        
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        
        # Find assistant tokens
        assistant_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        i = 0
        n = len(input_ids)
        
        while i < n:
            # Look for <|im_start|>assistant
            if input_ids[i] == assistant_start_id and i + 2 < n and input_ids[i+2] == self.tokenizer.convert_tokens_to_ids("assistant"):
                # Skip to content start
                j = i + 3
                if j < n and input_ids[j] == self.tokenizer.convert_tokens_to_id("\n"):
                    j += 1
                
                # Find corresponding <|im_end|>
                end_pos = None
                for k in range(j, n):
                    if input_ids[k] == assistant_end_id:
                        end_pos = k
                        break
                
                if end_pos is not None:
                    # Mark content as 1
                    for pos in range(j, end_pos):
                        if pos < len(mask):
                            mask[pos] = 1
                
                i = end_pos + 1 if end_pos is not None else n
            else:
                i += 1
        
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)