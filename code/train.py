import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import time
import warnings
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from k_model import ModelConfig, Transformer
from dataset import PretrainDataset, SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)


def get_lr(it, all, args):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch, train_loader, model, optimizer, scaler, ctx, args):
    start_time = time.time()
    iter_per_epoch = len(train_loader)
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

        if (step + 1) % args.save_interval == 0:
            model.eval()
            mode_name = "pretrain" if args.mode == "pretrain" else "sft"
            ckp = f'{args.save_dir}/{mode_name}_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'
            
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model(args, lm_config):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = Transformer(lm_config)
    
    if args.mode == "sft" and args.pretrained_path:
        Logger(f"Loading pretrained model from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    
    if args.gpus and ',' in args.gpus:
        num_gpus = len(args.gpus.split(','))
        if num_gpus > 1:
            Logger(f"Using {num_gpus} GPUs with DataParallel!")
            model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Training")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain", help="Training mode")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--data_path", type=str, required=True, help="Training data path")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer path")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Pretrained model path (for SFT)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--gpus", type=str, default='0', help="Comma-separated GPU IDs")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if torch.cuda.is_available():
            args.device = "cuda:0"

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
        vocab_size=6144,
        max_seq_len=args.max_seq_len
    )

    model, tokenizer = init_model(args, lm_config)
    
    if args.mode == "pretrain":
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    else:
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    Logger(f"Starting {args.mode} training...")
    for epoch in range(args.epochs):
        train_epoch(epoch, train_loader, model, optimizer, scaler, ctx, args)
    
    Logger("Training completed!")