# stats_stickers.py — какими стикерами я реально пользуюсь.
# Топ пойдёт в бота (этап 3): отвечать стикерами из моего набора.

import ijson
import json
from collections import Counter

try:
    from config import MY_FROM_ID, MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

IN_PATH = "result.json"
OUT_PATH = "stickers_top.json"

TOP_N = 25


def main():
    stickers = Counter()
    total = 0
    with open(IN_PATH, "rb") as f:
        for chat in ijson.items(f, "chats.list.item"):
            if chat.get("type") != "personal_chat":
                continue
            for msg in chat.get("messages", []):
                if msg.get("type") != "message":
                    continue
                if msg.get("from_id") != MY_FROM_ID:
                    continue
                if msg.get("media_type") != "sticker":
                    continue
                total += 1
                emoji = msg.get("sticker_emoji") or "?"
                # путь к файлу стикера пригодится, чтобы бот слал именно его
                stickers[(emoji, msg.get("file") or "")] += 1

    top = [
        {"emoji": e, "file": fpath, "count": c}
        for (e, fpath), c in stickers.most_common(TOP_N)
    ]
    print("Всего моих стикеров в личках:", total)
    for item in top:
        print(f"  {item['count']:>5} × {item['emoji']}  {item['file']}")

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        json.dump(top, out, ensure_ascii=False, indent=2)
    print("\nСохранено в", OUT_PATH)


if __name__ == "__main__":
    main()
