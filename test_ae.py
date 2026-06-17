import torch
from tokenizers import Tokenizer
from v30 import TextAutoencoder, config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = config["checkpoint_path"]
VOCAB_PATH = config["vocab_path"]

tokenizer = Tokenizer.from_file(VOCAB_PATH)

model = TextAutoencoder(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    num_latents=config["num_latents"],
    enc_layers=config["enc_layers"],
    dec_layers=config["dec_layers"]
).to(DEVICE)

# 載入權重（修正點）
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def test_fixed_length(text, target_len=256):
    input_ids = tokenizer.encode(text).ids
    # 裁切或填充到 target_len
    if len(input_ids) < target_len:
        input_ids += [0] * (target_len - len(input_ids))  # 假設 0 是 padding
    else:
        input_ids = input_ids[:target_len]
    
    x_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _ = model(x_tensor, target_len=target_len)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
    pred_text = tokenizer.decode(pred_ids)
    print("原始:", text[:50] + "..." if len(text)>50 else text)
    print("重建:", pred_text[:100] + "..." if len(pred_text)>100 else pred_text)

# 測試一段約 200~256 token 的文本
test_sentence = "問題：鳳梨酥是什麼？ 回覆：鳳梨酥是一種台灣的傳統糕點..."
test_fixed_length(test_sentence, target_len=256)
