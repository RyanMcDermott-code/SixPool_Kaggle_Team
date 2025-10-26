import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- Load the Base Model (again) ---
# You must load the same base model you trained on
model_name = "Qwen/Qwen2-7B-Instruct"
adapter_path = "llm_classification_finetuning/models/qwen_7B/final_dpo_adapter" # Path from our training script

# Load the base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   #Qwen models dont have pad token by default so we must set it.

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# --- Merge the LoRA adapter ---
# This loads your DPO-trained weights on top of the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

model.config.pad_token_id = tokenizer.pad_token_id

# You can also merge and unload to get a standalone model, 
# which is faster for inference:
# model = model.merge_and_unload()

print("✅ Model loaded and adapter merged.")

# --- Test the Model ---
prompt = "What is the best way to learn Python?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=150
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Model Response ---")
print(response)