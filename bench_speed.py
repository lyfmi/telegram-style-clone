# bench_speed.py — замер скорости генерации (токенов/сек) обученной LoRA.

import os
import time

from unsloth import FastModel

try:
    from config import MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(PROJECT_DIR, "lora_v2")

model, tokenizer = FastModel.from_pretrained(
    model_name=LORA_DIR, max_seq_length=1024, load_in_4bit=True
)
FastModel.for_inference(model)
tok = getattr(tokenizer, "tokenizer", tokenizer)
im_end = tok.convert_tokens_to_ids("<|im_end|>")

messages = [
    {"role": "system", "content": f"Ты — {MY_NAME}. Ты переписываешься в Telegram с человеком по имени Друг."},
    {"role": "user", "content": "расскажи подробно как прошел твой день сегодня"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

# прогрев (компиляция кэшей)
model.generate(**inputs, max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=im_end)

t0 = time.perf_counter()
out = model.generate(
    **inputs, max_new_tokens=128, min_new_tokens=128,
    do_sample=True, temperature=0.7, eos_token_id=None, pad_token_id=im_end,
)
dt = time.perf_counter() - t0
n = out.shape[1] - inputs["input_ids"].shape[1]
print(f"\n=== СКОРОСТЬ: {n} токенов за {dt:.1f} c = {n/dt:.1f} ток/с ===")
