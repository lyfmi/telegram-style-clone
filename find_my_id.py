# find_my_id.py — определяет твой from_id из экспорта Telegram.
#
# Логика: в личных чатах участвуют двое — ты и собеседник. Значит твой id
# встречается почти в каждом чате, а любой другой — только в своём.
#
# Запуск: python find_my_id.py   (рядом должен лежать result.json)

from collections import Counter

import ijson

IN_PATH = "result.json"

chats_with_id = Counter()
display_name = {}

print("Читаю result.json...")
with open(IN_PATH, "rb") as f:
    for chat in ijson.items(f, "chats.list.item"):
        if chat.get("type") != "personal_chat":
            continue
        ids_here = set()
        for msg in chat.get("messages", []):
            from_id = msg.get("from_id")
            if not from_id:
                continue
            ids_here.add(from_id)
            display_name.setdefault(from_id, msg.get("from") or "?")
        for fid in ids_here:
            chats_with_id[fid] += 1

top = chats_with_id.most_common(5)
if not top:
    raise SystemExit("В экспорте не нашлось личных чатов — проверь result.json")

print("\nКто в скольких личных чатах писал:")
for fid, n in top:
    print(f"  {n:>5} чатов   {fid}   ({display_name.get(fid)})")

best_id, best_n = top[0]
second_n = top[1][1] if len(top) > 1 else 0
print(f"\nТвой from_id, скорее всего: {best_id}")
if best_n < second_n * 3:
    print("!!! Разрыв с остальными небольшой — проверь глазами, "
          "что это действительно ты (по имени в скобках).")
print("Впиши его в config.py -> MY_FROM_ID")
