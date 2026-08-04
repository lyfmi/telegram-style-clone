#!/usr/bin/env python3
# sanitize_jsonl.py
import json
import re
from collections import Counter

# ---------- Regex patterns ----------
EMAIL_RE = re.compile(
    r'(?i)\b[a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,}\b'
)

# Phone numbers: tries to catch +country, spaces, dashes, parentheses
# Examples: +371 12345678, 8 800 555-35-35, (495)123-45-67
PHONE_RE = re.compile(
    r'(?x)(?<!\w)'
    r'(?:\+?\d{1,3}[\s\-()]*)?'          # optional country code
    r'(?:\(?\d{2,4}\)?[\s\-()]*)?'       # optional area code
    r'(?:\d[\s\-()]*){6,12}\d'           # main digits (7-13 total-ish)
    r'(?!\w)'
)

# IBAN: 2 буквы страны + 2 цифры + 11-30 буквоцифр (LV00XXXX0000000000000).
# Регекс телефона их не ловит — начинаются с букв.
IBAN_RE = re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b')

# JWT tokens (three base64url segments separated by dots)
JWT_RE = re.compile(r'\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b')

# Common token prefixes / auth headers / api keys
TOKEN_PREFIX_RE = re.compile(
    r'(?i)\b(?:bearer|token|apikey|api[_-]?key|secret|client[_-]?secret|access[_-]?token|refresh[_-]?token)\b\s*[:=]?\s*([A-Za-z0-9._\-]{12,})'
)

# "sk-..." style keys (e.g., many services)
SK_LIKE_RE = re.compile(r'\bsk-[A-Za-z0-9]{16,}\b')

# Long random strings that look like keys (base64-ish / hex-ish)
# We will mask only if long enough to likely be secret.
LONG_HEX_RE = re.compile(r'\b[a-f0-9]{32,}\b', re.IGNORECASE)
LONG_B64_RE = re.compile(r'\b[A-Za-z0-9+/]{32,}={0,2}\b')

# Codes / OTP: 4-8 digits near keywords
CODE_CONTEXT_RE = re.compile(
    r'(?i)\b(?:код|otp|2fa|одноразов|verification|verify|sms|смс|pin)\b[^0-9]{0,20}(\d{4,8})'
)

# Password patterns:
# 1) keywords near value: "пароль: qwerty", "password = 123", "pass  qwe"
PASSWORD_KV_RE = re.compile(
    r'(?i)\b(?:пароль|password|pass|pwd)\b\s*[:=]\s*([^\s]{4,})'
)

# 2) "мой пароль qwerty" style: keyword within 0-10 chars then a "value-like" token
PASSWORD_LOOSE_RE = re.compile(
    r'(?i)\b(?:пароль|password|pass|pwd)\b.{0,10}([^\s]{4,})'
)

# 3) If line contains strong password indicator words, we can mask whole line (optional)
STRONG_PASSWORD_HINT_RE = re.compile(
    r'(?i)\b(?:вот\s+пароль|мой\s+пароль|пароль\s+от|password\s+for|pass\s+for|логин\s+и\s+пароль)\b'
)

# ---------- Helpers ----------
def _safe_replace(pattern: re.Pattern, text: str, repl: str, counter: Counter, key: str) -> str:
    def _sub(m):
        counter[key] += 1
        return repl
    return pattern.sub(_sub, text)

def sanitize_text(text: str, stats: Counter) -> str:
    if not text:
        return text

    # If explicit "here is password ..." line: mask whole content (safer)
    if STRONG_PASSWORD_HINT_RE.search(text):
        stats["password_line"] += 1
        return "<PASSWORD_LINE>"

    # Emails
    text = _safe_replace(EMAIL_RE, text, "<EMAIL>", stats, "email")

    # Bank accounts (IBAN) — до токенов/телефонов, иначе разберут по кускам
    text = _safe_replace(IBAN_RE, text, "<IBAN>", stats, "iban")

    # Codes near keywords (mask just digits)
    def _code_sub(m):
        stats["code"] += 1
        # replace only captured digits
        return m.group(0).replace(m.group(1), "<CODE>")
    text = CODE_CONTEXT_RE.sub(_code_sub, text)

    # Password key/value (mask value)
    def _pw_kv_sub(m):
        stats["password_kv"] += 1
        return m.group(0).replace(m.group(1), "<PASSWORD>")
    text = PASSWORD_KV_RE.sub(_pw_kv_sub, text)
    # Loose password (less strict) — can false-positive; keep it after KV
    def _pw_loose_sub(m):
        stats["password_loose"] += 1
        return m.group(0).replace(m.group(1), "<PASSWORD>")
    text = PASSWORD_LOOSE_RE.sub(_pw_loose_sub, text)

    # Tokens/keys
    text = _safe_replace(JWT_RE, text, "<TOKEN>", stats, "jwt")
    text = _safe_replace(SK_LIKE_RE, text, "<TOKEN>", stats, "sk_like")

    # Token prefix patterns: keep keyword, mask value
    def _token_prefix_sub(m):
        stats["token_prefix"] += 1
        return m.group(0).replace(m.group(1), "<TOKEN>")
    text = TOKEN_PREFIX_RE.sub(_token_prefix_sub, text)

    # Long hex / base64-ish strings
    # NOTE: base64 regex can catch normal text fragments; we keep it but it's aggressive.
    text = _safe_replace(LONG_HEX_RE, text, "<TOKEN>", stats, "long_hex")
    text = _safe_replace(LONG_B64_RE, text, "<TOKEN>", stats, "long_b64")

    # Phones at the end to avoid messing with codes (digits)
    text = _safe_replace(PHONE_RE, text, "<PHONE>", stats, "phone")

    return text

def sanitize_jsonl(in_path: str, out_path: str) -> Counter:
    stats = Counter()
    bad_lines = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            msgs = obj.get("messages", [])
            if isinstance(msgs, list):
                for msg in msgs:
                    if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                        msg["content"] = sanitize_text(msg["content"], stats)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    stats["bad_lines"] = bad_lines
    return stats

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="train.jsonl", help="Input JSONL")
    p.add_argument("--out", dest="out", default="train_sanitized.jsonl", help="Output JSONL")
    args = p.parse_args()

    stats = sanitize_jsonl(args.inp, args.out)
    print("Done.")
    print("Stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")