import torch
from tokenizers import Tokenizer
from v30 import TextAutoencoder, config  # 假設你的主程式叫 v30.py

# ==================== 設定 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = config["checkpoint_path"]
VOCAB_PATH = config["vocab_path"]

print(f"🔍 載入 Autoencoder 測試環境 | 設備: {DEVICE}")

# 1. 載入 Tokenizer
tokenizer = Tokenizer.from_file(VOCAB_PATH)

# 2. 載入模型
model = TextAutoencoder(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    num_latents=config["num_latents"],
    enc_layers=config["enc_layers"],
    dec_layers=config["dec_layers"]
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print(f"✅ 成功載入權重: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 載入權重失敗: {e}")
    exit()

model.eval()

# ==================== 測試函數 ====================
def test_reconstruction(text):
    print("\n" + "="*50)
    print(f"📥 原始輸入: {text}")
    
    # 將文字轉為 Token IDs
    input_ids = tokenizer.encode(text).ids
    # 在非自迴歸中，我們通常讓輸出的長度等於輸入的長度來做精準還原
    target_len = len(input_ids) 
    
    x_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 通過 Encoder -> Compressor -> Decoder
            logits, _, _ = model(x_tensor, target_len=target_len)
            
    # 取機率最高的 Token (Greedy Decoding)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
    
    # 解碼回文字
    pred_text = tokenizer.decode(pred_ids)
    print(f"📤 模型還原: {pred_text}")
    print("="*50)

# ==================== 互動測試 ====================
if __name__ == "__main__":
    while True:
        user_input = input("\n請輸入一段文字讓模型壓縮並還原 (輸入 'q' 離開): ")
        if user_input.lower() == 'q':
            break
        test_reconstruction(user_input)