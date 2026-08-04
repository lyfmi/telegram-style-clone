# split_dataset.py — train/val сплит мульти-тёрн датасета (замена prepare_data.py).
# Старый фильтр word_count<=1 больше не нужен: короткие "ок"/"ага" в мульти-тёрн
# окнах идут с контекстом и как раз несут стиль.

import json
import random

INP = "dataset_sanitized.jsonl"
TRAIN_OUT = "train_v2.jsonl"
VAL_OUT = "val_v2.jsonl"

VAL_FRACTION = 0.03
VAL_MIN = 300

random.seed(42)

rows = []
with open(INP, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        msgs = obj.get("messages", [])
        has_assistant = any(m.get("role") == "assistant" and m.get("content") for m in msgs)
        has_user = any(m.get("role") == "user" and m.get("content") for m in msgs)
        if has_assistant and has_user:
            rows.append(obj)

random.shuffle(rows)

val_n = max(VAL_MIN, int(len(rows) * VAL_FRACTION))
val, train = rows[:val_n], rows[val_n:]

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(VAL_OUT, "w", encoding="utf-8") as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Всего:", len(rows))
print("Train:", len(train))
print("Val:", len(val))
