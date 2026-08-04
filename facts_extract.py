# facts_extract.py — этап 4 (RAG-память): досье фактов по контактам.
#
# По каждому контакту берёт последние сообщения из result.json, прогоняет через
# БАЗОВЫЙ Qwen3.5-4B (без LoRA — инструкциям следует лучше) и сохраняет выжимку
# в facts/<chat_id>.txt. Бот подставляет её в системный промпт.
# Досье — обычный текст: правь руками, бот перечитывает его на каждом ответе.
#
# ВНИМАНИЕ: нужен свободный GPU — останови serve_model.py перед запуском.
#
# Запуск: python facts_extract.py               # контакты из whitelist bot_config.json
#         python facts_extract.py --top 5       # плюс топ-5 чатов по моей активности
#         python facts_extract.py 123456789 ... # конкретные chat_id

import argparse
import json
import os
import re
from collections import Counter

from unsloth import FastModel

from sanitize_jsonl import sanitize_text  # тот же PII-маскинг, что у датасета

try:
    from config import MY_FROM_ID, MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

BASE_MODEL = "unsloth/Qwen3.5-4B"
RESULT_JSON = "result.json"
OUT_DIR = "facts"

MAX_MSGS = 600       # последних сообщений на контакт
CHUNK_CHARS = 8000   # ~2.5к токенов на кусок
MAX_CHUNKS = 6

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

EXTRACT_SYS = (
    "Ты помощник-аналитик. Тебе дают фрагменты переписки " + MY_NAME + " с его знакомым. "
    "Извлекай только факты, которые реально видны в тексте, без домыслов."
)
EXTRACT_TMPL = (
    "Фрагмент переписки " + MY_NAME + " с контактом «{name}»:\n\n{chunk}\n\n"
    "Выпиши маркированным списком (кратко, по-русски):\n"
    "1) как они обращаются друг к другу и в каком тоне общаются;\n"
    "2) факты о «{name}»: учёба/работа/город/увлечения/окружение;\n"
    "3) их общие темы, совместные дела, внутренние шутки;\n"
    "4) незакрытые темы последних разговоров.\n"
    "Только то, что есть в тексте. Если по пункту ничего нет — пропусти его."
)
MERGE_TMPL = (
    "Ниже несколько списков фактов из разных фрагментов переписки " + MY_NAME + " с «{name}». "
    "Объедини их в одно досье не длиннее 15 строк: убери повторы и мелочи, "
    "оставь самое характерное и полезное, чтобы писать этому человеку естественно. "
    "Начни ответ строкой «Что ты знаешь о {name}:» и дальше маркированный список.\n\n{notes}"
)


def normalize_text(t):
    if isinstance(t, list):
        return "".join(x if isinstance(x, str) else str(x.get("text", "")) for x in t)
    return t or ""


def media_placeholder(msg):
    mt = msg.get("media_type")
    if mt == "sticker":
        e = str(msg.get("sticker_emoji") or "").strip()
        return f"[стикер {e}]" if e else "[стикер]"
    named = {"voice_message": "[голосовое]", "video_message": "[кружок]",
             "animation": "[гифка]", "video_file": "[видео]", "audio_file": "[аудио]"}
    if mt in named:
        return named[mt]
    if "photo" in msg:
        return "[фото]"
    return None


def chat_lines(chat, name):
    """Последние MAX_MSGS сообщений чата в виде строк «Автор: текст»."""
    lines = []
    for m in chat.get("messages", []):
        if m.get("type") != "message" or not m.get("from_id"):
            continue
        text = normalize_text(m.get("text")).strip()
        if not text:
            text = media_placeholder(m) or ""
        if not text or len(text) > 800:
            continue
        who = MY_NAME if m["from_id"] == MY_FROM_ID else name
        # телефоны/карты/пароли не должны попасть ни в модель, ни в досье
        text = sanitize_text(text, Counter())
        lines.append(f"{who}: " + text.replace("\n", " "))
    return lines[-MAX_MSGS:]


def make_ask(model, tokenizer):
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def ask(system, user, max_new=450):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(prompt, str):
            prompt = prompt[0] if isinstance(prompt, (list, tuple)) and prompt else str(prompt)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.3, top_p=0.9,
            do_sample=True, eos_token_id=im_end, pad_token_id=im_end,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return THINK_RE.sub("", text).strip()

    return ask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ids", nargs="*", type=int, help="конкретные chat_id")
    p.add_argument("--top", type=int, default=0, help="добавить топ-N чатов по моей активности")
    args = p.parse_args()

    targets = set(args.ids)
    if not targets:  # без явных id — берём whitelist бота
        try:
            targets |= set(json.load(open("bot_config.json", encoding="utf-8")).get("whitelist", []))
        except FileNotFoundError:
            pass

    print("Читаю result.json...")
    data = json.load(open(RESULT_JSON, encoding="utf-8"))
    personal = [c for c in data["chats"]["list"] if c.get("type") == "personal_chat"]

    if args.top:
        def my_count(c):
            return sum(1 for m in c.get("messages", [])
                       if m.get("type") == "message" and m.get("from_id") == MY_FROM_ID)
        for c in sorted(personal, key=my_count, reverse=True)[:args.top]:
            targets.add(c["id"])

    by_id = {c["id"]: c for c in personal}
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Гружу базовую модель (без LoRA)...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=4096, load_in_4bit=True,
    )
    FastModel.for_inference(model)
    ask = make_ask(model, tokenizer)

    for cid in sorted(targets):
        out_path = os.path.join(OUT_DIR, f"{cid}.txt")
        chat = by_id.get(cid)
        name = (chat.get("name") or "знакомый").strip() if chat else "знакомый"
        if not chat or len(chat.get("messages", [])) < 20:
            if os.path.exists(out_path):
                print(f"[{cid}] {name}: истории нет, {out_path} уже существует — не трогаю")
                continue
            # истории нет (например, свежий тестовый аккаунт) — заготовка для ручного заполнения
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Что ты знаешь о {name}:\n- (истории мало — впиши факты руками)\n")
            print(f"[{cid}] {name}: истории нет, создал заготовку {out_path}")
            continue

        lines = chat_lines(chat, name)
        text = "\n".join(lines)
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)][-MAX_CHUNKS:]
        print(f"[{cid}] {name}: {len(lines)} сообщений, {len(chunks)} кусков...")

        notes = []
        for i, chunk in enumerate(chunks, 1):
            notes.append(ask(EXTRACT_SYS, EXTRACT_TMPL.format(name=name, chunk=chunk)))
            print(f"  кусок {i}/{len(chunks)} готов")
        dossier = notes[0] if len(notes) == 1 else ask(
            EXTRACT_SYS, MERGE_TMPL.format(name=name, notes="\n\n---\n\n".join(notes)), max_new=600)
        if not dossier.startswith("Что ты знаешь"):
            dossier = f"Что ты знаешь о {name}:\n" + dossier
        dossier = sanitize_text(dossier, Counter())  # страховка на выходе

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dossier + "\n")
        print(f"  -> {out_path}")

    print("Готово. Проверь и поправь досье руками — это обычный текст.")


if __name__ == "__main__":
    main()
