# Tiny-Text-Latent-Diffusion (TTLDM)

> **代號：D2-Diffuser / v30** > 一個專為消費級顯示卡（12GB VRAM）設計的 12M 超微型文字潛在擴散模型實驗。探索「雙向狀態空間（Bi-Mamba）+ 潛在壓縮 + 非自迴歸平行解碼」的全新文字生成路徑。

---

## 📖 專案概述 (Project Overview)

傳統的自迴歸（Autoregressive）Transformer 面臨兩大核心瓶頸：
1. **二次複雜度的注意力機制**，導致上下文長度受限，難以擴展至超長文本。
2. **逐 Token 自迴歸生成**，解碼速度受限，且無法像繪圖 AI 一樣「一口氣」平行產出完整內容。

**TTLDM (v30)** 的核心目標是**打穿一條全新的技術路徑**：
我們徹底放棄逐字接龍，改用 **Bi-Mamba 雙向狀態空間模型** 作為編碼骨幹，透過 **Perceiver-style 交叉注意力** 將可變長度的文本序列，極致壓縮至一組固定大小的**潛在向量（Latent Vectors）**。最後，由**非自迴歸（Non-Autoregressive）解碼器**在同一時間步並行吐出所有 Token，實現真正的「一口氣生成」。

本專案是一個約 **12M 參數** 的微型驗證模型，旨在證明該技術路徑在消費級硬體上的可行性與高效率。

---

## ✨ 核心特色 (Key Features)

* **雙向狀態空間（Bi-Mamba Encoder）**：每一層由正向與反向 Mamba 塊組成，提供線性時間與空間複雜度的序列建模，擺脫位置編碼的束縛。
* **極致潛在壓縮（Context Compressor）**：使用 64 個可學習的查詢向量（Queries），將長文本壓縮為固定 `64 × 256` 的潛在空間，大幅降低後續擴散去噪的計算開銷。
* **潛在空間正規化（Latent Regularization）**：在 Autoencoder 階段引入 KL 散度懲罰（KL Loss），強制將潛在空間塑造成標準常態分佈球體，為第二階段的擴散去噪（Diffusion Prior）鋪平道路。
* **非自迴歸並行解碼（NAR Decoder）**：解碼器完全不使用 Causal Mask，徹底打破自迴歸枷鎖，生成速度不受輸出文本長度影響。
* **極致的硬體優化**：
  * **數學向量化（Vectorized Cumsum）**：將 Mamba 的時間序列循環運算全部改寫為對數空間的並行累積和（Cumsum），消滅 Python 迴圈瓶頸，帶來 **30+ 倍的 GPU 加速**。
  * **AMP 自動混合精度**：全面支援 `torch.amp` 的 `autocast` 與 `GradScaler`，完美壓榨 12GB VRAM 顯示卡的極限效能。

---

## 🏗️ 模型架構 (Architecture)


```

輸入文本 (Token IDs)
│
▼
Embedding + Bi-Mamba Encoder (4 層)
│
▼
ContextCompressor (Cross-Attention, 64 個可學習查詢向量)
│
▼
潛在向量 z (固定大小 64 × 256) ─── [加入 KL 散度鎖定常態分佈]
│
▼
Non-Autoregressive Latent Decoder (4 層 Transformer Decoder)
│
▼
輸出文本 (Token IDs 一口氣並行平行輸出)



---

## ⚙️ 超參數設定 (Hyperparameters)

| 參數 | 數值 | 說明 |
| :--- | :--- | :--- |
| `vocab_size` | 16385 | BPE Tokenizer v13 (含煞車控制符 `<|endoftext|>`) |
| `d_model` | 256 | 全模型統一隱藏維度 |
| `num_latents` | 64 | 壓縮後的固定潛在向量數量 |
| `enc_layers` | 4 | Bi-Mamba 編碼器層數 |
| `dec_layers` | 4 | Transformer NAR 解碼器層數 |
| `block_size` | 256 | 訓練時的序列長度上限 |
| `batch_size` | 8 | 消費級硬體黃金 Batch 大小 |
| `lr` | 3e-4 | 針對非自迴歸架構優化的平穩學習率 |
| `kl_weight` | 1e-4 | KL Loss 懲罰權重 |

---

## 📅 訓練計畫 (Training Roadmap)

### 📈 階段 1：Autoencoder 預訓練（文字重建）
* **目的**：讓編碼器、壓縮器與非自迴歸解碼器學會將文本壓縮成 64 個向量，並能精準還原回原文。
* **損失函數**：`Total_Loss = CrossEntropy(Reconstruction) + 1e-4 * KL_Loss`
* **執行指令**：
  ```bash
  python v30.py



### 📉 階段 2：潛在空間擴散訓練（Diffusion Prior）—— *Coming Soon*

* **目的**：凍結階段 1 的 AE 權重，訓練一個 1D 卷積 UNet 機器，在 `64 × 256` 的常態分佈潛在空間上進行逐步去噪（DDPM/DDIM 排程），學習從純高斯雜訊中生成有意義的文字隱向量。

---

## 🚀 聯絡與交流 (Contributing)

本專案是一次針對文字生成新範式的激進探索。如果你對**狀態空間模型（SSM）**、非自迴歸解碼（NAR）**或**文本擴散（Text Diffusion）有任何想法、優化建議或實驗數據，歡迎開 Issue 或提交 PR 一起交流！


