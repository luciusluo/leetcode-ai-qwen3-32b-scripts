from accelerate import Accelerator
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import logging
from datetime import datetime


# Global
MODEL_PATH = "/hy-tmp/model_hub/qwen/Qwen3-32B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # device_map="auto"
)


# === 加载并预处理数据 ===
def preprocess(example):
    prompt = f"用户：{example['instruction']}\n助手：{example['output']}"
    tokenized = TOKENIZER(prompt, truncation=True, max_length=3072)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# === 自定义 Callback：打印 loss ===
class LossCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            train_loss = logs.get("loss")
            if train_loss is not None:
                logger.info(f"[Step {state.global_step}] ✅ Train Loss: {train_loss:.4f}")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            logger.info(f"[Step {state.global_step}] 🔍 Eval  Loss: {eval_loss:.4f}")



def main(train_data_path: str, valid_data_path: str, model_output_dir: str, log_output_dir:str):
    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用LoRA
    model = get_peft_model(MODEL, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": train_data_path, "validation": valid_data_path})
    train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset["validation"].map(preprocess, remove_columns=dataset["validation"].column_names)

    # === 训练参数 ===
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_strategy="epoch",
        # 学习率和优化器
        learning_rate=1e-6,
        warmup_ratio=0.05,
        optim="adamw_torch",
        weight_decay=0.1,
        adam_beta1=0.9, # ✅ 1阶动量
        adam_beta2=0.95,   # ✅ 2阶动量
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=True,
        # 其他步数
        logging_strategy="steps",
        logging_dir=log_output_dir,
        logging_steps=5,
        eval_steps=5,
        eval_strategy="no",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

    # === 初始化 Trainer ===
    trainer = Trainer(
        model=model,
        tokenizer=TOKENIZER,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LossCallback()]
    )

    trainer.train()

    # === 保存微调后的模型（LoRA 权重） ===
    trainer.model.save_pretrained(model_output_dir)
    TOKENIZER.save_pretrained(model_output_dir)

    print(f"LoRA 微调完成，模型已保存至：{model_output_dir}")





if __name__ == "__main__":
    train_data_path = "/home/workspace/Qwen3-32b/corpus/0.train/dsr_ershen_20250619.json"
    valid_data_path = "/home/workspace/Qwen3-32b/corpus/1.valid/anhao_to_annoed_data_qwen3_format.json"
    model_output_dir = "/hy-tmp/exps/qwen3-32b/lora-1"
    log_output_dir = os.path.join(model_output_dir, "logs")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    log_file = os.path.join(log_output_dir, f"train_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger("train_logger")

    main(train_data_path, valid_data_path, model_output_dir, log_output_dir)