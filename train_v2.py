# train_v2.py — QLoRA-дообучение Qwen3.5-4B на train_v2.jsonl (этап 2 плана).
# Запускается ИЗНУТРИ WSL Ubuntu через train.sh.
#
# Ключевые отличия от train_qlora.py первой части:
#   - Unsloth (быстрее и экономнее по VRAM на 8GB карте)
#   - loss считается ТОЛЬКО на моих репликах (train_on_responses_only)
#   - чекпоинты каждые SAVE_STEPS шагов + автопродолжение с последнего
#     (остановил Ctrl+C — просто запусти train.sh ещё раз)
#   - early stopping по eval_loss, в конце берётся лучший чекпоинт
#
# Чекпоинты лежат в ~/tgstyle/out (ext4 — быстрее, переживает перезапуск WSL),
# финальная LoRA копируется в папку проекта (lora_v2/), видна из Windows.

import os
import shutil

from unsloth import FastModel  # импорт unsloth должен идти до transformers
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Qwen3.5-4B"
MAX_SEQ_LEN = 1024

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(PROJECT_DIR, "train_v2.jsonl")
VAL_FILE = os.path.join(PROJECT_DIR, "val_v2.jsonl")
FINAL_LORA_DIR = os.path.join(PROJECT_DIR, "lora_v2")

OUT_DIR = os.path.expanduser("~/tgstyle/out")

SAVE_STEPS = 50  # чекпоинт ~каждые полчаса; при остановке теряется максимум этот кусок

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

dataset = load_dataset(
    "json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE}
)


def to_text(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }


dataset = dataset.map(to_text, remove_columns=dataset["train"].column_names)

print("=== Пример отрендеренного текста (проверка шаблона) ===")
print(dataset["train"][0]["text"][:1500])
print("========================================================")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=OUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        num_train_epochs=1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=True,  # чистый лог: строки с loss вместо прогресс-бара
        report_to="none",
        seed=42,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

# loss только на моих репликах (между <|im_start|>assistant и <|im_end|>)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

last_ckpt = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
if last_ckpt:
    print(f">>> Продолжаю с чекпоинта: {last_ckpt}")
else:
    print(">>> Старт с нуля")

trainer.train(resume_from_checkpoint=last_ckpt)

# лучший (по eval_loss) адаптер — в папку проекта, видную из Windows
model.save_pretrained(FINAL_LORA_DIR)
tokenizer.save_pretrained(FINAL_LORA_DIR)
print("Готово. LoRA сохранена в:", FINAL_LORA_DIR)
