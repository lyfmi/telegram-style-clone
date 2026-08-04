# stats_timing.py — распределение задержек моих ответов по часам суток.
# Результат (timing_profile.json) использует бот на этапе 3,
# чтобы отвечать с реалистичными паузами.

import ijson
import json
from datetime import datetime
from statistics import median

try:
    from config import MY_FROM_ID, MY_NAME
except ImportError:
    raise SystemExit(
        "Нет config.py — скопируй пример и впиши свои данные:\n"
        "  cp config.example.py config.py"
    )

IN_PATH = "result.json"
OUT_PATH = "timing_profile.json"

MAX_DELAY = 4 * 3600  # ответы позже 4 часов не считаем "ответом в диалоге"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def main():
    # delays[hour] = задержки (сек) моих ответов, отправленных в этот час
    delays = {h: [] for h in range(24)}

    with open(IN_PATH, "rb") as f:
        for chat in ijson.items(f, "chats.list.item"):
            if chat.get("type") != "personal_chat":
                continue
            partner_last_t = None
            answered = True
            for msg in chat.get("messages", []):
                if msg.get("type") != "message":
                    continue
                from_id = msg.get("from_id")
                if not from_id:
                    continue
                try:
                    t = int(msg.get("date_unixtime"))
                except (TypeError, ValueError):
                    continue
                if from_id == MY_FROM_ID:
                    if not answered and partner_last_t is not None:
                        delay = t - partner_last_t
                        if 0 <= delay <= MAX_DELAY:
                            hour = datetime.fromtimestamp(t).hour
                            delays[hour].append(delay)
                        answered = True
                else:
                    partner_last_t = t
                    answered = False

    profile = {}
    print(f"{'час':>4} {'ответов':>8} {'медиана':>9} {'p25':>7} {'p75':>7}")
    for h in range(24):
        vals = sorted(delays[h])
        if vals:
            profile[str(h)] = {
                "count": len(vals),
                "median_sec": round(median(vals)),
                "p25_sec": percentile(vals, 0.25),
                "p75_sec": percentile(vals, 0.75),
            }
            print(
                f"{h:>4} {len(vals):>8} {round(median(vals)):>8}с "
                f"{percentile(vals, 0.25):>6}с {percentile(vals, 0.75):>6}с"
            )
        else:
            profile[str(h)] = {"count": 0}
            print(f"{h:>4} {0:>8}      —       —       —")

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        json.dump(profile, out, ensure_ascii=False, indent=2)
    print("\nСохранено в", OUT_PATH)


if __name__ == "__main__":
    main()
