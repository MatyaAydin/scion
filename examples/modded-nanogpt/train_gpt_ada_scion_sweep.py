# Modified from: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from datargs import parse

from ada_scion import AdaScion


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = (math.sqrt(2)*F.relu(x)).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    # optimization hyperparams
    batch_size : int = 8*64
    device_batch_size : int = 64
    sequence_length : int = 1024
    num_iterations : int = 1500
    warmup_iters : int = 75
    warmdown_iters : int = 250
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 75
    val_tokens : int = 10485760
    save_every : int = 0
    # model hyperparams
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    # optimizer hyperparams
    unconstrained : bool = False
    momentum : float = 0.9
    scale : float = 50
    last_scale : float = 300
    # which hyperparameter to sweep (set via CLI: --sweep lr|momentum|eps|power_frequency|beta)
    sweep : str = 'lr'


def main(args, optim_args):
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0)

    if master_process:
        print("======== Arguments ========")
        print(args)
        print(f"optim_args: {optim_args}")
        print("===========================")

    B, T = args.device_batch_size, args.sequence_length
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    x, y = train_loader.next_batch()

    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    optim_groups = [{
        'params': raw_model.transformer.h.parameters(),
        'norm': 'Spectral',
        'norm_kwargs': {'steps': 5},
        'scale': args.scale,
    }, {
        'params': raw_model.lm_head.parameters(),
        'norm': 'Sign',
        'norm_kwargs': {},
        'scale': args.last_scale,
    }]
    optimizer1 = AdaScion(optim_groups, unconstrained=args.unconstrained, **optim_args)
    optimizers = [optimizer1]

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    if master_process:
        os.makedirs("logs_adascion", exist_ok=True)
        study_name = f"logs_ada_scion_lr_{optim_args['lr']}_momentum_{optim_args['momentum']}_beta_{optim_args['beta_eucl']}_eps_{optim_args['eps']}_powerfreq_{optim_args['power_frequency']}_{optim_args['order']}"
        logfile = f"logs_adascion/{study_name}.txt"
        with open(logfile, "w") as f:
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()
    train_loader.reset()
    train_loss_history = np.zeros(args.num_iterations+1)

    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1

        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        model.train()
        for i in range(1, train_accumulation_steps+1):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
                train_loss_history[step] += train_loss.item() / train_accumulation_steps
            x, y = train_loader.next_batch()
            if i < train_accumulation_steps:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        for p in model.parameters():
            p.grad /= train_accumulation_steps

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f}")
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f}\n")

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    return train_loss_history[:args.num_iterations]


if __name__ == "__main__":
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    args = parse(Hyperparameters)

    ddp_rank = int(os.environ['RANK'])
    master_process = (ddp_rank == 0)

    DEFAULTS = {
        "lr":              5e-5,
        "momentum":        0.9,
        "beta":            0.999,  # applied to both beta_eucl and beta_spectral
        "eps":             1e-2,
        "power_frequency": 100,
    }

    sweep_values = {
        "lr":              list(np.logspace(-6, -4, 10)),
        "momentum":        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "beta":            [0.85, 0.9, 0.99, 0.999],
        "eps":             list(np.logspace(-4, 0, 5)),
        "power_frequency": [25, 50, 100, 250, 500],
    }

    assert args.sweep in sweep_values, \
        f"--sweep must be one of {list(sweep_values.keys())}, got '{args.sweep}'"

    for value in sweep_values[args.sweep]:
        # Build optim_args from defaults, overriding the swept param.
        # beta controls both beta_eucl and beta_spectral together.
        d = {**DEFAULTS, args.sweep: value}
        optim_args = {
            "lr":                    d["lr"],
            "momentum":              d["momentum"],
            "beta_eucl":             d["beta"],
            "beta_spectral":         d["beta"],
            "use_trace_normalization": True,
            "power_frequency":       int(d["power_frequency"]),
            "eps":                   d["eps"],
            "order": 8
        }
        if master_process:
            print(f"\n{'='*60}")
            print(f"Sweeping {args.sweep} = {value}")
            print(f"optim_args: {optim_args}")
            print(f"{'='*60}\n")
        main(args, optim_args)

    dist.destroy_process_group()
