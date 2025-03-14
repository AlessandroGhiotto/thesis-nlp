import os
import sys

# Get the absolute path to the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src._utils._generate_dataset import main_generate_dataset


#############################################
# LOAD MODEL
#############################################

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

#############################################
# GENERATE BASELINE AGNEWS DATASET
#############################################

prompt = f"""\
You are an expert in journalism and NLP specializing in news classification. \
Your task is to generate 10 high-quality short documents, that talks about the following four News categories:  
- **Business**
- **Sci/Tech**
- **Sports**
- **World**

### **Output Format (JSON)**  
Return only a valid JSON list of 10 items in the following structure:

```json
[
    {{"text": <text>, "label": <label>}},
    ...
]
```
"""

config = {
    "model": model,
    "tokenizer": tokenizer,
    "generation_method": "baseline",
    "prompt": prompt,
    "system_prompt": None,
    "num_examples": 500,
    "max_new_tokens": 4096,
    "seed": 42,
    "json_output_file": "synthetic_data/datasets/syn_agnews_baseline_500.json",
    "log_file": "src/agnews/generate_dataset_agnews_log.json",
}
main_generate_dataset(config)

#############################################
# GENERATE TARGETED AGNEWS DATASET
#############################################

prompt = f"""\
You are an expert in journalism and NLP specializing in news classification. \
Your task is to generate 10 high-quality short documents, that talks about the following four News categories (labels):  
- **Business**
- **Sci/Tech**
- **Sports**
- **World**

For each example, also list the key phenomena it covers.

### **Follow these topics:**
- **Business**  
  - Markets  
  - Economy  
  - Companies  
  - Startups  
  - Regulations  

- **Sci/Tech**  
  - AI  
  - Space  
  - Cybersecurity  
  - Biotech  
  - Climate  

- **Sports**  
  - Events  
  - Records  
  - Highlights  
  - Scandals  
  - Olympics  

- **World**  
  - Politics  
  - Conflicts  
  - Disasters  
  - Human Rights  
  - Trade

### **Output Format (JSON)**
The labels must be one of the specified categories, which are: Business, Sci/Tech, Sports, World.
Return only a valid JSON list of 10 elements in the following structure:

```json
[
    {{"text": <text of the document>, "label": <corresponding label>, "phenomena": ["<phenomenon1>", "<phenomenon2>", ...]}},
    ...
]
```
"""

config = {
    "model": model,
    "tokenizer": tokenizer,
    "generation_method": "targeted + linguistic tags",
    "prompt": prompt,
    "system_prompt": None,
    "num_examples": 500,
    "max_new_tokens": 4096,
    "seed": 42,
    "json_output_file": "synthetic_data/datasets/syn_agnews_targeted+tags_500.json",
    "log_file": "src/agnews/generate_dataset_agnews_log.json",
}
main_generate_dataset(config)
