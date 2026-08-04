# test_chat.py — консольный чат с обученной LoRA (smoke-тест после train_v2.py).
# Запуск из WSL:  source ~/tgstyle/.venv/bin/activate && python test_chat.py [имя собеседника]
#
# Можно писать НЕСКОЛЬКО сообщений подряд (как в телеге):
#   пустая строка — отправить всё накопленное, /q — выход.
# Ответ стримится по мере генерации.

import json
import os
import re
import sys

from unsloth import FastModel
from transformers import TextStreamer

try:
    from config import MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean_reply(text):
    """Убирает блоки размышлений, включая незакрытые (обрезанные лимитом токенов),
    чтобы они не попадали в историю диалога и не портили следующие ответы."""
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # блок был открыт ещё в промпте
        text = text.split("</think>")[-1]
    if "<think>" in text:  # блок открыт, но не закрыт — дальше только размышления
        text = text.split("<think>")[0]
    return text.strip()


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(PROJECT_DIR, "lora_v2")
MAX_SEQ_LEN = 1024

SYSTEM_TMPL = (
    "Ты — " + MY_NAME + ". Ты переписываешься в Telegram с человеком по имени {name}. "
    "Отвечай коротко и естественно, так, как обычно пишешь в личных сообщениях."
)

contact = sys.argv[1] if len(sys.argv) > 1 else "Друг"

model, tokenizer = FastModel.from_pretrained(
    model_name=LORA_DIR,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
FastModel.for_inference(model)

tok = getattr(tokenizer, "tokenizer", tokenizer)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
if im_end is None or im_end == tok.unk_token_id:
    im_end = tok.eos_token_id
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

messages = [{"role": "system", "content": SYSTEM_TMPL.format(name=contact)}]
print(f"\nЧат от лица: {MY_NAME}, собеседник: {contact}.")
print("Пиши сообщения построчно; ПУСТАЯ строка — отправить, /q — выход.\n")

while True:
    lines = []
    while True:
        try:
            line = input(f"{contact}> ")
        except (EOFError, KeyboardInterrupt):
            line = "/q"
        if line.strip() == "/q":
            sys.exit(0)
        if line.strip() == "":
            if lines:
                break
            continue  # пустая строка без накопленных сообщений — ждём дальше
        lines.append(line.strip())

    messages.append({"role": "user", "content": "\n".join(lines)})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # пустой <think></think>, как в обучающих данных
    )
    if not isinstance(prompt, str):
        # редкий сбой обёртки процессора: сохраняем улики и чинимся на лету
        with open(os.path.join(PROJECT_DIR, "chat_debug.json"), "w", encoding="utf-8") as dbg:
            json.dump({"prompt_type": str(type(prompt)), "prompt_repr": repr(prompt)[:2000],
                       "messages": messages}, dbg, ensure_ascii=False, indent=2)
        print(f"[warn: шаблон вернул {type(prompt).__name__}, улики в chat_debug.json]")
        prompt = prompt[0] if isinstance(prompt, (list, tuple)) and prompt else str(prompt)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    print(f"{MY_NAME}: ", end="", flush=True)
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.6,   # рекомендация Qwen для не-думающего режима:
        top_p=0.8,         # меньше словесного салата, чем 0.7/0.9
        repetition_penalty=1.05,
        do_sample=True,
        eos_token_id=im_end,
        pad_token_id=im_end,
        streamer=streamer,
    )
    prompt_len = inputs["input_ids"].shape[1]
    reply = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    reply = clean_reply(reply)
    messages.append({"role": "assistant", "content": reply})
    print()

    # не даём контексту расти бесконечно
    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]
