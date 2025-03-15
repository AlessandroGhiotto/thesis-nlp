# 🛠️ Utilities

This folder contains utility scripts for managing synthetic dataset generation.

---

## 📄 Files Overview

- **`_helpers_.py`**  
  Helper functions that are reused in the other scripts and the notebooks

- **`_generate_dataset.py`**  
  Script for generating a dataset, given a configuration. It is imported and executed in the folder corresponding to a dataset.

---

### `_generate_dataset.py` Example Usage

```python
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src._utils._generate_dataset import main

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
    # if I want to add other model parameters, I can add them here
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

prompt = "example prompt"

# EXAMPLE CONFIG DICT
config = {
    "dataset": "dataset_name",
    "model": model,  # THE ACTUAL MODEL OBJECT
    "tokenizer": tokenizer,  # THE ACTUAL TOKENIZER OBJECT
    "generation_method": "targeted",
    "prompt": prompt,
    "system_prompt": None,
    "num_examples": 500,  # number of examples to be generated
    "max_new_tokens": 8192,  # per generation call (not total)
    "seed": 42,
    "json_output_file": "synthetic_data/datasets/example.json",
    "log_file": "src/semevalirony/example_log.json",
    "correct_labels": ["positive", "negative"],
    "correct_fields": ["text", "label"]
}
main_generate_dataset(config)
```
