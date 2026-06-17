import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.amp import autocast, GradScaler  # 🚀 新增混合精度模組

# ==================== 基礎組件 ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        hidden_dim = int(d_model * 8 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.ln = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x_norm = self.ln(x)
        out = self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))
        return self.dropout(out)

# ==================== 簡化 Mamba 塊 (單向) ====================
class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, expand=2, d_state=16, kernel_size=4):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)

        self.ln = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=kernel_size, groups=self.d_inner, padding=kernel_size-1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        xz = self.in_proj(x_norm)
        x_hidden, z = xz.chunk(2, dim=-1)

        x_conv = x_hidden.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        dtA = dt.unsqueeze(-1) * A  # (B, L, D_inner)
        
        # 🚀 向量化加速術：利用 Cumsum 在並行空間完成 RNN 運算
        cum_dtA = torch.cumsum(dtA, dim=1) 
        safe_div = torch.exp(cum_dtA) + 1e-8  # 移除錯誤的 unsqueeze
        
        dtB = dt.unsqueeze(-1) * B_mat.unsqueeze(-2) 
        dtB_x = dtB * x_conv.unsqueeze(-1) 
        
        div_x = dtB_x / safe_div
        states = torch.cumsum(div_x, dim=1) * torch.exp(cum_dtA)  # 移除錯誤的 unsqueeze
        
        y = (states * C_mat.unsqueeze(-2)).sum(dim=-1)
        y = y + x_conv * self.D
        out = F.silu(z) * y
        return x + self.out_proj(out)

# ==================== 雙向 Mamba 塊 ====================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, expand=2, d_state=16):
        super().__init__()
        self.forward_mamba = SimpleMambaBlock(d_model, expand, d_state)
        self.backward_mamba = SimpleMambaBlock(d_model, expand, d_state)
        self.ffn = SwiGLU(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x_f = self.forward_mamba(x)
        x_b = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        x = x + x_f + x_b
        x = x + self.ffn(self.norm(x))  
        return x

# ==================== 上下文壓縮器 ====================
class ContextCompressor(nn.Module):
    def __init__(self, d_model, num_latents=64, num_heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, encoder_output):
        B, L, D = encoder_output.shape
        queries = self.latent_queries.expand(B, -1, -1)  # [B, K, D]
        latents = self.cross_attn(query=queries, key=encoder_output, value=encoder_output)[0]
        latents = self.norm1(latents + queries)
        latents = self.norm2(latents + self.ffn(latents))
        return latents  # [B, K, D]

# ==================== 非自迴歸 Transformer 解碼器 ====================
class NonAutoregressiveLatentDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, d_model=256, n_layers=4, n_heads=8, max_len=2048):
        super().__init__()
        self.max_len = max_len
        self.pos_embed = nn.Embedding(max_len, d_model)
        # 不使用 Causal Mask，達成一口氣並行生成
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                                       batch_first=True, dropout=0.1) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, latents, target_len):
        B, K, D = latents.shape
        positions = torch.arange(0, target_len, device=latents.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.pos_embed(positions)  # [B, L, D]
        
        for layer in self.layers:
            tgt_emb = layer(tgt_emb, memory=latents)
            
        logits = self.output_proj(self.final_norm(tgt_emb))
        
        # 🚀 修復：嚴格限制 logits 的數值範圍，防止初期預測極端化導致 Loss 飆高/NaN
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        return logits

# ==================== 完整 Autoencoder (含 KL 正規化) ====================
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_latents=64, enc_layers=4, dec_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 🚀 初始化修復：將權重標準差縮小到 0.02，避免點積爆炸
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        self.encoder_blocks = nn.ModuleList([BiMambaBlock(d_model) for _ in range(enc_layers)])
        self.compressor = ContextCompressor(d_model, num_latents)
        
        # 🚀 同步修復壓縮器的 Query 初始化
        nn.init.normal_(self.compressor.latent_queries, mean=0.0, std=0.02)
        
        self.decoder = NonAutoregressiveLatentDecoder(
            latent_dim=d_model, 
            vocab_size=vocab_size, 
            d_model=d_model, 
            n_layers=dec_layers, 
            max_len=2048
        )
        self.decoder.output_proj.weight = self.embedding.weight

    def forward(self, input_ids, target_len=None):
        x = self.embedding(input_ids)
        for block in self.encoder_blocks:
            x = block(x)
            
        latents = self.compressor(x)  # [B, K, d_model]
        
        # 為 Latent Space 計算 KL Divergence，確保它是一個漂亮的球體
        latent_mean = latents.mean(dim=-1) 
        latent_var = latents.var(dim=-1)   
        kl_loss = 0.5 * torch.mean(latent_mean**2 + latent_var - torch.log(latent_var.clamp(min=1e-6)) - 1)
        
        if target_len is not None:
            logits = self.decoder(latents, target_len)
            return logits, latents, kl_loss
        else:
            return latents  

# ==================== 設定參數與執行環境 ====================
config = {
    "vocab_path": "bpe_tokenizer_v13.json", 
    "bin_path": "v30.bin",
    "vocab_size": 16385,
    "d_model": 256,
    "num_latents": 64,
    "enc_layers": 4,
    "dec_layers": 4,
    "batch_size": 8,
    "block_size": 256, 
    "lr": 3e-4,          # 🚀 修復：學習率調降，防止初期暴走
    "kl_weight": 1e-4, 
    "epochs": 100,
    "checkpoint_path": "autoencoder_stage1_nar.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 V30 Autoencoder 啟動 | 設備: {device}")

try:
    tokenizer = Tokenizer.from_file(config["vocab_path"])
    config["vocab_size"] = tokenizer.get_vocab_size()
    if os.path.exists(config["bin_path"]):
        data = np.memmap(config["bin_path"], dtype=np.uint16, mode='r')
        print(f"✅ 成功載入語料，大小: {len(data)} tokens")
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"⚠️ 找不到 Tokenizer 或 Bin 檔，使用隨機張量進行架構測試...")
    data = np.random.randint(0, config["vocab_size"], size=(100000,), dtype=np.uint16)

# ==================== 補全的 Batch 函數 ====================
def get_batch(block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), x.pin_memory().to(device, non_blocking=True)

# ==================== Stage 1: 基礎訓練迴圈 ====================
def train_stage1():
    model = TextAutoencoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_latents=config["num_latents"],
        enc_layers=config["enc_layers"],
        dec_layers=config["dec_layers"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # 🚀 修復：加入混合精度 Scaler 來大幅提升訓練速度
    scaler = GradScaler('cuda') 
    
    max_steps = 50000  
    pbar = tqdm(range(max_steps), desc="訓練 AE")
    
    for step in pbar:
        x, y_target = get_batch(config["block_size"], config["batch_size"])
        
        optimizer.zero_grad()
        
        # 🚀 修復：使用 autocast 開啟混合精度計算 (如果顯卡支援 bfloat16 最佳，否則預設 float16)
        with autocast('cuda', dtype=torch.bfloat16):
            logits, latents, kl_loss = model(x, target_len=config["block_size"])
            recon_loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y_target.view(-1))
            total_loss = recon_loss + config["kl_weight"] * kl_loss
        
        # 🚀 修復：使用 Scaler 進行反向傳播與梯度更新
        scaler.scale(total_loss).backward()
        
        # 梯度裁剪需要在 unscale 之後進行
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if step % 10 == 0:
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.3f}",
                "Recon": f"{recon_loss.item():.3f}",
                "KL": f"{kl_loss.item():.4f}"
            })
            
        if step > 0 and step % 1000 == 0:
            torch.save(model.state_dict(), config["checkpoint_path"])

if __name__ == "__main__":
    train_stage1()