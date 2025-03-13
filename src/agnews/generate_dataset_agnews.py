import os
import sys

# Get the absolute path to the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src._utils._generate_dataset import main

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    attn_implementation="flash_attention_2",
    # if I want to add other model parameters, I can add them here
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

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

# Example configuration dictionary
config = {
    "model": model,  # THE ACTUAL MODEL OBJECT
    "tokenizer": tokenizer,  # THE ACTUAL TOKENIZER OBJECT
    "generation_method": "targeted",
    "prompt": prompt,
    "system_prompt": None,
    "num_examples": 500,
    "max_new_tokens": 4096,  # per generation call (not total)
    "seed": 42,
    "json_output_file": "synthetic_data/datasets/syn_agnews_targeted_500.json",
    "log_file": "src/agnews/dataset_generation_log.json",
}
main(config)
