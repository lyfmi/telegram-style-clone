# bot_v2.py — подключение модели к ЛИЧНОМУ аккаунту через Telegram Business (этап 3).
# Это не отдельный чат с ботом: модель отвечает ЗА МЕНЯ в моих личках.
#
# Настройка (один раз):
#   1. @BotFather -> /newbot -> токен -> в bot_config.json (скопировать из bot_config.example.json)
#   2. @BotFather -> /mybots -> выбрать бота -> Bot Settings -> Business Mode -> Turn on
#   3. Телефон: Настройки -> Telegram для бизнеса -> Чат-боты -> подключить бота
#   4. Написать себе с другого аккаунта: бот залогирует user_id -> внести в whitelist
#
# Запуск (после serve_model.py): source ~/tgstyle/.venv/bin/activate && python bot_v2.py
#
# Поведенческий слой:
#   - копит серию входящих сообщений и отвечает один раз (как человек, дочитавший всё)
#   - задержка ответа — из timing_profile.json по текущему часу (p25..p75 + поправка на длину)
#   - «печатает…» перед каждым сообщением, ответ сплитится по \n на отдельные сообщения
#   - ночью (night_hours) молчит; изредка игнорит короткие реплики (ignore_prob)
#   - [стикер X] от модели -> реальный стикер из stickers_top.json
#   - входящие медиа -> те же плейсхолдеры, что при обучении ([стикер 😂], [фото], ...)
#   - если Я сам ответил вручную — запланированный автоответ отменяется

import asyncio
import json
import logging
import random
import re
import time
from collections import defaultdict, deque
from datetime import datetime

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot_v2")

try:
    CFG = json.load(open("bot_config.json", encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(
        "Нет bot_config.json. Создай его из примера и вставь токен от @BotFather:\n"
        "  cp bot_config.example.json bot_config.json\n"
        "  nano bot_config.json   # поле \"token\""
    )
if not CFG.get("token") or "ВСТАВЬ" in CFG["token"]:
    raise SystemExit("В bot_config.json не вставлен токен от @BotFather (поле \"token\").")
TIMING = json.load(open("timing_profile.json", encoding="utf-8"))
STICKERS = json.load(open("stickers_top.json", encoding="utf-8"))

STICKER_RE = re.compile(r"^\[стикер\s*(\S*)\]$")
PLACEHOLDER_RE = re.compile(r"^\[[^\]]+\]")  # [фото], [голосовое]... — модель их слать не должна

histories = defaultdict(lambda: deque(maxlen=CFG.get("history_max", 30)))
pending = defaultdict(list)   # накопленные необработанные входящие
reply_tasks = {}              # chat_id -> отложенная задача ответа
deadlines = {}                # chat_id -> monotonic-время, когда отвечаем
recent_replies = defaultdict(lambda: deque(maxlen=5))  # анти-залипание: последние ответы


def incoming_to_text(msg):
    """Входящее сообщение -> текст с теми же плейсхолдерами, что при обучении."""
    cap = f" {msg.caption}" if msg.caption else ""
    if msg.text:
        text = msg.text
    elif msg.sticker:
        emoji = (msg.sticker.emoji or "").strip()
        text = f"[стикер {emoji}]" if emoji else "[стикер]"
    elif msg.voice:
        text = "[голосовое]"
    elif msg.video_note:
        text = "[кружок]"
    elif msg.photo:
        text = "[фото]" + cap
    elif msg.animation:  # проверять ДО document: гифка — тоже document
        text = "[гифка]" + cap
    elif msg.video:
        text = "[видео]" + cap
    elif msg.audio:
        text = "[аудио]" + cap
    elif msg.document:
        text = "[файл]" + cap
    elif msg.location:
        text = "[геопозиция]"
    elif msg.contact:
        text = "[контакт]"
    else:
        text = "[сообщение]"
    if msg.forward_origin is not None:
        text = f"[переслал] {text}" if text else "[пересланное сообщение]"
    # собеседник ответил на конкретное сообщение — показываем модели, на какое,
    # иначе «кто это» в ответ на старую реплику читается как вопрос из ниоткуда
    quoted = ""
    if msg.quote is not None and msg.quote.text:  # выделенная цитата из сообщения
        quoted = msg.quote.text
    elif msg.reply_to_message is not None:
        r = msg.reply_to_message
        quoted = (r.text or r.caption or "").strip()
        if not quoted:
            quoted = ("стикер" if r.sticker else "фото" if r.photo
                      else "голосовое" if r.voice else "сообщение")
    if quoted:
        quoted = quoted.replace("\n", " ")
        if len(quoted) > 60:
            quoted = quoted[:57] + "..."
        text = f"[в ответ на «{quoted}»] {text}"
    return text


def reply_delay():
    """Пауза перед ответом: реальное распределение из timing_profile по часу суток."""
    prof = TIMING.get(str(datetime.now().hour))
    if not prof:
        return random.uniform(5, 20)
    lo, hi = prof.get("p25_sec", 3), prof.get("p75_sec", 30)
    delay = random.uniform(lo, min(hi, lo + 60))  # хвост p75 бывает в минуты — режем
    return delay * CFG.get("delay_scale", 1.0)


def typing_time(text):
    """Сколько «печатать» сообщение: ~5 символов в секунду, 1..7 с."""
    return min(max(len(text) / 5.0, 1.0), 7.0)


def pick_sticker(emoji):
    """Стикер по эмодзи из топа; нет совпадения — случайный, взвешенный по частоте."""
    matched = [s for s in STICKERS if s["emoji"] == emoji]
    pool = matched or STICKERS
    return random.choices(pool, weights=[s["count"] for s in pool], k=1)[0]["file"]


def load_system_extra(chat_id):
    """RAG-память: досье контакта (facts/<chat_id>.txt) + «что у меня сейчас»
    (me_now.txt). Перечитывается на каждом ответе — можно править на лету."""
    parts = []
    for path in (f"facts/{chat_id}.txt", "me_now.txt"):
        try:
            content = open(path, encoding="utf-8").read().strip()
            if content:
                if path == "me_now.txt":
                    content = "Что у тебя сейчас происходит:\n" + content
                parts.append(content)
        except FileNotFoundError:
            pass
    return "\n\n".join(parts)


async def ask_model(contact, history, chat_id, temp_bump=0.0):
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(CFG.get("server_url", "http://127.0.0.1:8008/reply"),
                              json={"contact": contact, "messages": list(history),
                                    "system_extra": load_system_extra(chat_id),
                                    "temperature": CFG.get("temperature", 0.6) + temp_bump,
                                    "top_p": CFG.get("top_p", 0.8)})
        r.raise_for_status()
        return r.json()["reply"]


def reply_lines(reply):
    """Строки ответа без подряд идущих дублей («я на машине» ×6 -> ×1), максимум 6."""
    lines = []
    for line in (l.strip() for l in reply.split("\n")):
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return lines[:6]


async def deliver_reply(chat_id, bcid, context):
    """Сработавший таймер: спросить модель и отправить ответ по-человечески."""
    reply_tasks.pop(chat_id, None)
    batch, pending[chat_id] = pending[chat_id], []
    if not batch:
        return

    hour = datetime.now().hour
    if hour in CFG.get("night_hours", [5, 6, 7, 8]):
        log.info("ночь (%d ч) — молчу для %s", hour, chat_id)
        return
    total_text = "\n".join(m["text"] for m in batch)
    if len(total_text) <= 5 and random.random() < CFG.get("ignore_prob", 0.07):
        log.info("игнорю короткое от %s (как живой человек)", chat_id)
        return

    contact = batch[0]["contact"]
    try:
        reply = await ask_model(contact, histories[chat_id], chat_id)
        # анти-залипание: то же самое уже говорил недавно — пробуем погорячее,
        # если и это повтор — молчим (пауза естественнее заевшей пластинки)
        norm = " ".join(reply.split()).lower()
        if norm and norm in recent_replies[chat_id]:
            log.info("повтор недавнего ответа (%r) — регенерирую", norm[:40])
            reply = await ask_model(contact, histories[chat_id], chat_id, temp_bump=0.25)
            norm = " ".join(reply.split()).lower()
            if norm in recent_replies[chat_id]:
                log.info("снова повтор — молчу")
                return
    except Exception:
        log.exception("сервер модели недоступен — молчу (не палимся ошибками в чат)")
        return
    if not reply:
        return
    recent_replies[chat_id].append(norm)

    for line in reply_lines(reply):
        m = STICKER_RE.match(line)
        if m:
            try:
                with open(pick_sticker(m.group(1)), "rb") as f:
                    await context.bot.send_sticker(
                        chat_id=chat_id, sticker=f, business_connection_id=bcid)
                histories[chat_id].append({"role": "assistant", "content": line})
            except Exception:
                log.exception("не смог отправить стикер, пропускаю")
            continue
        if PLACEHOLDER_RE.match(line):
            log.info("модель выдала плейсхолдер %r — пропускаю", line)
            continue
        await context.bot.send_chat_action(
            chat_id=chat_id, action=ChatAction.TYPING, business_connection_id=bcid)
        await asyncio.sleep(typing_time(line))
        await context.bot.send_message(
            chat_id=chat_id, text=line, business_connection_id=bcid)
        histories[chat_id].append({"role": "assistant", "content": line})
        log.info("-> %s: %s", chat_id, line)


async def on_business_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.business_message
    if msg is None or msg.chat is None or msg.from_user is None:
        return
    chat_id = msg.chat.id
    text = incoming_to_text(msg)

    # в бизнес-чате chat.id == id собеседника; мои собственные сообщения приходят
    # с from_user == я — их кладём в историю и отменяем автоответ (я уже сам в чате)
    if msg.from_user.id != chat_id:
        histories[chat_id].append({"role": "assistant", "content": text})
        task = reply_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            pending[chat_id] = []
            log.info("ответил вручную в %s — автоответ отменён", chat_id)
        return

    if chat_id not in CFG.get("whitelist", []):
        log.info("НЕ в whitelist: id=%s (%s %s @%s) — «%s»", chat_id,
                 msg.from_user.first_name or "", msg.from_user.last_name or "",
                 msg.from_user.username or "-", text[:60])
        return

    contact = CFG.get("names", {}).get(str(chat_id)) or msg.from_user.first_name or "знакомый"
    histories[chat_id].append({"role": "user", "content": text})
    pending[chat_id].append({"contact": contact, "text": text})
    log.info("<- %s (%s): %s", contact, chat_id, text[:80])

    # первое сообщение задаёт паузу из профиля; следующие лишь чуть сдвигают
    # дедлайн (ждём, не допишет ли), а не перезапускают всю паузу заново
    now = time.monotonic()
    quiet_gap = max(3.0, 8.0 * CFG.get("delay_scale", 1.0))
    if deadlines.get(chat_id, 0) > now:
        deadlines[chat_id] = max(deadlines[chat_id], now + quiet_gap)
    else:
        deadlines[chat_id] = now + reply_delay()
    delay = deadlines[chat_id] - now

    old = reply_tasks.pop(chat_id, None)
    if old:
        old.cancel()

    async def delayed():
        try:
            await asyncio.sleep(delay)
            deadlines.pop(chat_id, None)
            await deliver_reply(chat_id, msg.business_connection_id, context)
        except asyncio.CancelledError:
            pass

    reply_tasks[chat_id] = asyncio.create_task(delayed())


def main():
    app = Application.builder().token(CFG["token"]).build()
    app.add_handler(MessageHandler(filters.ALL, on_business_message))
    log.info("Бот запущен. Whitelist: %s", CFG.get("whitelist"))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
