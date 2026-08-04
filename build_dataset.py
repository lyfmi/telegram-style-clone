# build_dataset.py — экспорт Telegram (result.json) -> мульти-тёрн датасет (dataset.jsonl)
#
# Отличия от старого main.py:
#   - только личные чаты (personal_chat), группы/каналы/боты не попадают в датасет
#   - мульти-тёрн: окна до MAX_TURNS реплик с чередованием ролей user/assistant
#   - серия подряд идущих сообщений одного автора склеивается через "\n"
#     (при инференсе бот сплитит ответ по "\n" и шлёт отдельными сообщениями)
#   - в каждом примере системный промпт с именем собеседника
#   - текст не нормализуется агрессивно: регистр, пунктуация и переносы сохраняются
#   - МОИ пересланные сообщения выкидываются (это не мой текст)
#   - медиа и пересылки СОБЕСЕДНИКА не выкидываются, а заменяются плейсхолдерами
#     ([стикер 😂], [фото], [голосовое], [переслал] ...) — иначе мои ответы на них
#     выглядят ответами невпопад (~10% ответов); бот делает ту же подстановку
#   - из моих медиа остаются только стикеры ([стикер 😂]) — бот умеет их отправлять

import ijson
import json
import re

try:
    from config import MY_FROM_ID, MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

IN_PATH = "result.json"
OUT_PATH = "dataset.jsonl"


SESSION_GAP = 4 * 3600      # разрыв больше 4 часов = новый диалог
BURST_GAP = 300             # тот же автор после паузы >5 мин = новый диалог
                            # (иначе монологи за день склеиваются в один гигантский ход)
MAX_TURN_MSGS = 6           # максимум сообщений в одном ходе (защита от простыней)
MAX_MSG_LINES = 10          # сообщения-«пасты» (списки, логи, код) — не разговорный
MAX_MSG_CHARS = 1200        # стиль, выкидываем целиком
MAX_TURNS = 10              # максимум реплик (ходов) в одном примере
MAX_CHARS = 3000            # ограничение длины примера (~1000 токенов)
MIN_MY_TURNS = 1            # в окне должна быть хотя бы одна моя реплика
MIN_PARTNER_TURNS = 1       # и хотя бы одна реплика собеседника

SYSTEM_TMPL = (
    "Ты — " + MY_NAME + ". Ты переписываешься в Telegram с человеком по имени {name}. "
    "Отвечай коротко и естественно, так, как обычно пишешь в личных сообщениях."
)

multi_nl_re = re.compile(r"\n{3,}")
spaces_re = re.compile(r"[ \t]+")


def normalize_text(t):
    """Собирает текст из строки или списка сущностей экспорта.
    Сохраняет регистр, пунктуацию и одиночные переносы строк."""
    if t is None:
        return ""
    if isinstance(t, str):
        s = t
    elif isinstance(t, list):
        parts = []
        for x in t:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and "text" in x:
                parts.append(str(x["text"]))
        s = "".join(parts)
    else:
        s = str(t)
    s = s.replace("\r", "")
    s = spaces_re.sub(" ", s)
    s = multi_nl_re.sub("\n\n", s)
    return s.strip()


def media_placeholder(msg):
    """Короткая текстовая пометка вместо медиа: [стикер 😂], [фото], [голосовое]…
    None, если сообщение не медиа."""
    mt = msg.get("media_type")
    if mt == "sticker":
        emoji = str(msg.get("sticker_emoji") or "").strip()
        return f"[стикер {emoji}]" if emoji else "[стикер]"
    named = {
        "voice_message": "[голосовое]",
        "video_message": "[кружок]",
        "animation": "[гифка]",
        "video_file": "[видео]",
        "audio_file": "[аудио]",
    }
    if mt in named:
        return named[mt]
    if "photo" in msg:
        return "[фото]"
    if "contact_information" in msg:
        return "[контакт]"
    if "location_information" in msg:
        return "[геопозиция]"
    if "file" in msg:
        return "[файл]"
    return None


def iter_messages(chat):
    """Выдаёт (is_me, text, unixtime) для содержательных сообщений чата.

    Медиа собеседника не выкидывается, а помечается плейсхолдером — иначе
    мои ответы на стикеры/фото выглядят в датасете ответами невпопад.
    Из МОИХ медиа остаются только стикеры: их бот умеет отправлять,
    а обещать «сейчас скину фото» модель не должна."""
    for msg in chat.get("messages", []):
        if msg.get("type") != "message":
            continue
        from_id = msg.get("from_id")
        if not from_id:
            continue
        is_me = from_id == MY_FROM_ID
        forwarded = msg.get("forwarded_from") is not None
        if is_me and forwarded:
            continue  # мои пересылки — не мой текст
        text = normalize_text(msg.get("text"))
        if text and (text.count("\n") + 1 > MAX_MSG_LINES or len(text) > MAX_MSG_CHARS):
            continue  # вставленные списки/логи/код
        ph = media_placeholder(msg)
        if is_me:
            if not text:
                if ph and ph.startswith("[стикер"):
                    text = ph
                else:
                    continue  # моё фото/голосовое без подписи — не для датасета
        else:
            if ph:
                text = f"{ph} {text}".strip() if text else ph
            if forwarded:
                text = f"[переслал] {text}" if text else "[переслал сообщение]"
            if not text:
                continue
        try:
            t = int(msg.get("date_unixtime"))
        except (TypeError, ValueError):
            continue
        yield (is_me, text, t)


def split_sessions(messages):
    """Режет поток сообщений на сессии.
    Разрыв: пауза > SESSION_GAP, либо тот же автор продолжает после паузы
    > BURST_GAP (монолог с перерывами — не один ход, а разные диалоги)."""
    session = []
    prev_t = None
    prev_is_me = None
    for m in messages:
        is_me, _text, t = m
        gap = None if prev_t is None else t - prev_t
        if gap is not None and (
            gap > SESSION_GAP or (is_me == prev_is_me and gap > BURST_GAP)
        ):
            if session:
                yield session
            session = []
        session.append(m)
        prev_t = t
        prev_is_me = is_me
    if session:
        yield session


def to_turns(session):
    """Склеивает подряд идущие сообщения одного автора в один ход через \\n.
    Не больше MAX_TURN_MSGS сообщений на ход — лишнее отбрасывается."""
    turns = []  # (is_me, text, n_msgs)
    for is_me, text, _t in session:
        if turns and turns[-1][0] == is_me:
            if turns[-1][2] < MAX_TURN_MSGS:
                turns[-1] = (is_me, turns[-1][1] + "\n" + text, turns[-1][2] + 1)
        else:
            turns.append((is_me, text, 1))
    return [(is_me, text) for is_me, text, _n in turns]


def to_windows(turns):
    """Режет список ходов на непересекающиеся окна.
    Хвост из чужих реплик переносится в следующее окно,
    чтобы каждое окно заканчивалось моим ответом."""
    windows = []
    cur = []
    cur_chars = 0
    for turn in turns:
        cur.append(turn)
        cur_chars += len(turn[1])
        if len(cur) >= MAX_TURNS or cur_chars >= MAX_CHARS:
            carry = []
            while cur and not cur[-1][0]:  # снимаем чужие реплики с конца
                carry.insert(0, cur.pop())
            if cur:
                windows.append(cur)
            cur = carry
            cur_chars = sum(len(t[1]) for t in cur)
    while cur and not cur[-1][0]:
        cur.pop()
    if cur:
        windows.append(cur)
    return windows


def window_to_sample(window, contact_name):
    my_turns = sum(1 for t in window if t[0])
    partner_turns = len(window) - my_turns
    if my_turns < MIN_MY_TURNS or partner_turns < MIN_PARTNER_TURNS:
        return None
    messages = [{"role": "system", "content": SYSTEM_TMPL.format(name=contact_name)}]
    for is_me, text in window:
        messages.append({"role": "assistant" if is_me else "user", "content": text})
    return {"messages": messages, "contact": contact_name}


def main():
    stats = {"chats": 0, "sessions": 0, "windows": 0, "samples": 0, "my_turns": 0}
    with open(IN_PATH, "rb") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for chat in ijson.items(f, "chats.list.item"):
            if chat.get("type") != "personal_chat":
                continue
            name = normalize_text(chat.get("name")) or "знакомый"
            msgs = list(iter_messages(chat))
            if not msgs:
                continue
            stats["chats"] += 1
            for session in split_sessions(msgs):
                stats["sessions"] += 1
                for window in to_windows(to_turns(session)):
                    stats["windows"] += 1
                    sample = window_to_sample(window, name)
                    if sample:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        stats["samples"] += 1
                        stats["my_turns"] += sum(
                            1 for m in sample["messages"] if m["role"] == "assistant"
                        )
    print("Личных чатов с текстом:", stats["chats"])
    print("Сессий:", stats["sessions"])
    print("Окон:", stats["windows"])
    print("Примеров в датасете:", stats["samples"])
    print("Моих реплик (assistant-ходов):", stats["my_turns"])


if __name__ == "__main__":
    main()
