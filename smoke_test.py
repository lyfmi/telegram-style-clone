# smoke_test.py — быстрая неинтерактивная проверка обученной LoRA:
# несколько типовых сообщений -> ответы модели в консоль.

import os
import re

from unsloth import FastModel

try:
    from config import MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean_reply(text):
    """Убирает блоки размышлений, включая незакрытые (обрезанные лимитом токенов)."""
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # блок был открыт ещё в промпте
        text = text.split("</think>")[-1]
    if "<think>" in text:  # блок открыт, но не закрыт — дальше только размышления
        text = text.split("<think>")[0]
    return text.strip()


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(PROJECT_DIR, "lora_v2"))

SYSTEM_TMPL = (
    "Ты — " + MY_NAME + ". Ты переписываешься в Telegram с человеком по имени {name}. "
    "Отвечай коротко и естественно, так, как обычно пишешь в личных сообщениях."
)

PROBES = [
    ("Друг", "привет, чё делаешь?"),
    ("Друг", "го завтра встретимся?"),
    ("Мама", "Сынок, ты покушал сегодня?"),
    ("Друг", "видел новый ролик про нейронки?"),
    ("Одногруппник", "ты идешь завтра на пары?"),
    # медиа-плейсхолдеры — модель должна реагировать на них как на медиа
    ("Друг", "[стикер 😂]"),
    ("Друг", "[фото]\nсмотри какой сетап собрал"),
    ("Друг", "[голосовое]"),
]

model, tokenizer = FastModel.from_pretrained(
    model_name=LORA_DIR,
    max_seq_length=1024,
    load_in_4bit=True,
)
FastModel.for_inference(model)

for contact, text in PROBES:
    messages = [
        {"role": "system", "content": SYSTEM_TMPL.format(name=contact)},
        {"role": "user", "content": text},
    ]
    # процессор Qwen3.5 в tokenize-режиме ждёт VLM-формат сообщений,
    # поэтому рендерим шаблон в строку (как при обучении) и токенизируем сами
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # пустой <think></think>, как в обучающих данных
    )
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    # стоп строго на конце реплики, иначе модель играет и за собеседника
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end == tok.unk_token_id:
        im_end = tok.eos_token_id
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.6,   # рекомендация Qwen для не-думающего режима:
        top_p=0.8,         # меньше словесного салата, чем 0.7/0.9
        repetition_penalty=1.05,
        do_sample=True,
        eos_token_id=im_end,
        pad_token_id=im_end,
    )
    prompt_len = inputs["input_ids"].shape[1]
    reply = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    reply = clean_reply(reply)
    print(f"\n[{contact}] {text}")
    for part in reply.split("\n"):
        if part.strip():
            print(f"  >> {part.strip()}")
