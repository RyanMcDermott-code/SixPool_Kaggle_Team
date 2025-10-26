import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import torch

print(torch.cuda.is_available())

def remap_data_for_training(csv):
    csv_data = pd.read_csv(csv)
    train = []

    for index, row in csv_data.iterrows():
        train_row = {
            'id': row["id"],
            'prompt': row["prompt"],
        }

        if row["winner_model_a"] == 1:
            train_row['chosen'] = row['response_a']
            train_row['rejected'] = row['response_b']
        elif row["winner_model_b"] == 1:
            train_row['chosen'] = row['response_b']
            train_row['rejected'] = row['response_a']

        train.append(train_row)

    data = pd.DataFrame(train)

    # Ensure our df types are strings
    data['prompt'] = data['prompt'].astype(str)
    data['chosen'] = data['chosen'].astype(str)
    data['rejected'] = data['rejected'].astype(str)

    return data

csv_path = "llm_classification_finetuning/data/train.csv"
df_train = remap_data_for_training(csv_path)
ds_train = Dataset.from_pandas(df_train)

model_name = 'Qwen/Qwen2.5-7B-Instruct'

# Using QLoRA configuration (4-bit) to squeeze the model down to fit into 4090
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   #Qwen models dont have pad token by default so we must set it.

# Load the SFT model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

# Defind the PEFT LoRA configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
)

# DPO Trainer config
training_args = DPOConfig(
    output_dir="llm_classification_finetuning/models/qwen_7B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=2,
    save_strategy="epoch",
    report_to="none",
    beta=0.1
)

# wrap the model with PEFT
model = get_peft_model(model, peft_config)

# Initialize trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=ds_train,
    processing_class=tokenizer,
)

print("Starting DPO Training...")

dpo_trainer.train()

print("DPO training finished")

# Save the final LoRA adapter
final_model_path = "llm_classification_finetuning/models/qwen_7B/final_dpo_adapter"
dpo_trainer.save_model(final_model_path)

print(f"LoRA adapter saved to {final_model_path}")