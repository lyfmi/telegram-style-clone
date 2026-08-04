# serve_model.py — локальный HTTP-сервер инференса для телеграм-бота (этап 3).
# Держит lora_v2 в VRAM, бот ходит сюда за ответами.
#
# Запуск: source ~/tgstyle/.venv/bin/activate && python serve_model.py
#
# POST /reply {"contact": "Саня", "messages": [{"role": "user", "content": "го в зал"}, ...]}
#          -> {"reply": "не могу\nу меня вождение"}
#
# Пока transformers (~14 ток/с — для коротких реплик с человеческими задержками
# хватает); переезд на llama.cpp/GGUF — отдельным шагом, бот не изменится.

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from unsloth import FastModel

try:
    from config import MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

HOST, PORT = "127.0.0.1", 8008
MAX_SEQ_LEN = 1024
HISTORY_MAX = 20  # реплик контекста (обучались на окнах до 10)

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

SYSTEM_TMPL = (
    "Ты — " + MY_NAME + ". Ты переписываешься в Telegram с человеком по имени {name}. "
    "Отвечай коротко и естественно, так, как обычно пишешь в личных сообщениях."
)

print("Загружаю модель...")
model, tokenizer = FastModel.from_pretrained(
    model_name="lora_v2", max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
)
FastModel.for_inference(model)
tok = getattr(tokenizer, "tokenizer", tokenizer)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
if IM_END is None or IM_END == tok.unk_token_id:
    IM_END = tok.eos_token_id
GEN_LOCK = threading.Lock()  # GPU одна — генерации строго по очереди


def clean_reply(text):
    """Убирает блоки размышлений, включая незакрытые (обрезанные лимитом токенов)."""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def merge_roles(history):
    """Склеивает подряд идущие сообщения одной роли через \n —
    как в обучающем датасете (чередование ролей обязательно)."""
    merged = []
    for m in history:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    return merged


def generate_reply(contact, history, system_extra="", temperature=0.6, top_p=0.8):
    system = SYSTEM_TMPL.format(name=contact)
    if system_extra:  # RAG-память: досье контакта + «что у меня сейчас» (этап 4)
        system += "\n\n" + system_extra
    messages = [{"role": "system", "content": system}]
    messages += merge_roles(history)[-HISTORY_MAX:]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # пустой <think></think>, как в обучающих данных
    )
    if not isinstance(prompt, str):  # редкий сбой обёртки процессора
        prompt = prompt[0] if isinstance(prompt, (list, tuple)) and prompt else str(prompt)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with GEN_LOCK:
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            do_sample=True,
            eos_token_id=IM_END,
            pad_token_id=IM_END,
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_reply(reply)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/reply":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = generate_reply(body.get("contact") or "знакомый", body["messages"],
                                   body.get("system_extra", ""),
                                   float(body.get("temperature", 0.6)),
                                   float(body.get("top_p", 0.8)))
            data = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:  # один плохой запрос не должен ронять сервер
            print("ошибка запроса:", e)
            self.send_error(500, str(e))

    def log_message(self, fmt, *args):
        pass  # не спамим access-логом


if __name__ == "__main__":
    print(f"Сервер готов: http://{HOST}:{PORT}/reply")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
