import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
import json
import glob

@dataclass
class RWKVConfig:
    vocab_size = 65536
    n_embd = 2048
    n_layer = 24
    dim_att = 2048
    dim_ffn = 7168
    n_head = 32
    head_size = 64
    head_size_divisor = 8

# TODO: Add roll support to mlx to avoid this conversion
def time_shift(x: mx.array):
    tx = np.array(x.astype(mx.float32))
    cn = np.roll(tx, 1, axis=x.ndim-2)
    mask = np.ones(tx.shape[-1])
    mask[0] = 0
    mask = np.expand_dims(mask, mask.ndim)
    return mx.array(cn * mask).astype(x.dtype)

class RWKVChannelMix(nn.Module):
    def __init__(self, config: RWKVConfig):
        self.time_mix_k = mx.zeros((1, 1, config.n_embd))
        self.time_mix_r = mx.zeros((1, 1, config.n_embd))
        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)
    
    def __call__(self, x:mx.array):
        xx = time_shift(x) # x.pad((0, 0, 1, -1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = nn.relu(k) ** 2
        kv = self.value(k)
        return mx.sigmoid(self.receptance(xr)) * kv

class RWKVTimeMix(nn.Module):
    def __init__(self, config: RWKVConfig):
        self.n_head = config.n_head
        self.head_size_divisor = config.head_size_divisor
        self.time_mix_k = mx.zeros((1, 1, config.n_embd))
        self.time_mix_v = mx.zeros((1, 1, config.n_embd))
        self.time_mix_r = mx.zeros((1, 1, config.n_embd))
        self.time_mix_g = mx.zeros((1, 1, config.n_embd))
        
        self.time_decay = mx.zeros((config.n_head, config.head_size))
        self.time_faaa = mx.zeros((config.n_head, config.head_size))
        
        self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.key = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.value = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.output = nn.Linear(config.dim_att, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(config.n_head, config.dim_att)
        
    def __call__(self, x: mx.array, sx: mx.array, s: mx.array):
        # _N_ = head_size
        B, T, C = x.shape
        H = self.n_head
        # add a row of zeros to the top and remove the last row
        # this is essentially a shift down. mlx does not support negative pad
        xx = time_shift(x) # x.pad((0, 0, 1, -1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = nn.silu(self.gate(xg))

        # cuda kernel here
        s = mx.zeros((self.n_head))
        out = mx.zeros((T, H, C // H), dtype=x.dtype)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:, :, t:t+1]
            vt = v[:, t:t+1, :]
            at = mx.matmul(kt, vt)
            out[t] = mx.squeeze((rt @ (self.time_faaa * at + s)), 1)
            s = at + self.time_decay * s
        x = out.reshape((T, C)) 
        x = x.reshape((B * T, C))
        x = self.ln_x(x / self.head_size_divisor).reshape((B, T, C))
        return self.output(x * g)
        

class Block(nn.Module):
    def __init__(self, index: int, config: RWKVConfig):
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = RWKVTimeMix(config)
        self.ffn = RWKVChannelMix(config)

    def __call__(self, x: mx.array, sx: mx.array, s: mx.array):
        ax, sx, s = self.attn(self.ln1(x), sx, s)
        x = x + ax
        return x + self.ffn(self.ln2(x)), sx, x



class RWKV(nn.Module):
    def __init__(self, config: RWKVConfig):
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(i, config) for i in range(config.n_layer)]
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ln0 = nn.LayerNorm(config.n_embd)
        self.state = [None] * config.n_layer * 3
        for i in range(config.n_layer):
            self.state[i*3+0] = mx.zeros(config.n_embd)
            self.state[i*3+1] = mx.zeros((config.n_head, config.dim_att // config.n_head, config.dim_att // config.n_head))
            self.state[i*3+2] = mx.zeros(config.n_embd)

    @classmethod
    def load_weights(cls, path: str):
        model_args = RWKVConfig()
        with open(f"{path}/config.json", "r") as f:
            config = json.loads(f.read())
            model_args.dim_att = config["attention_hidden_size"]
            model_args.head_size = config["head_size"]
            model_args.n_layer = config["num_hidden_layers"]
            model_args.vocab_size = config["vocab_size"]
            model_args.n_embd = config["hidden_size"]

        res = RWKV(model_args)
        weight_files = glob.glob(f"{path}/*.safetensors")
        if len(weight_files) != 1:
            raise ValueError(f"No weight files found in {path}")
        weight = mx.load(weight_files[0])
        for k, v in weight.items():
            if k.startswith("blocks.0.ln0"):
                k.replace("blocks.0.ln0", "ln0")
                setattr(res, k, v)
            else:
                setattr(res, k, v)
        return res

    def __call__(self, x: mx.array):
        x = self.emb(x)
        x = self.ln0(x)
        for i, block in enumerate(self.blocks):
            x, self.state[i*3+0], self.state[i*3+1] = block(x, self.state[i*3+0], self.state[i*3+1])
        x = self.ln_out(x)
        return self.head(x)
    
if __name__ == "__main__":
    model = RWKV.load_weights("./rwkv5-world-1b")