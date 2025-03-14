import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from src._utils._helpers import response2json, get_response, set_seed, clear_cuda_cache


def log_generation(details, log_file):
    """Log the generation details to a JSON file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(details)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
    print(f"Logged generation to {log_file}")


def save_dataset_json(metadata, output_file):
    """Save the dataset along with metadata in a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Dataset with metadata saved to {output_file}")


# def generate_synthetic_data_rolling(
#     prompt_template,
#     num_examples,
#     model,
#     tokenizer,
#     batch_size=50,
#     max_new_tokens=8192,
#     system_prompt=None,
# ):
#     """
#     Generate synthetic data in batches using a rolling context.
#     The model is expected to output a JSON array of examples.

#     Each example should have at least 'text' and 'label' keys,
#     and optionally a 'phenomena' key.

#     If a system_prompt is provided, the function uses the chat template via
#     tokenizer.apply_chat_template.

#     Returns:
#         List[dict]: A list of generated examples.
#     """

#     def build_prompt(previous_examples):
#         """
#         Build a prompt with rolling context using the last few examples.
#         """
#         if previous_examples:
#             # Include the last 5 examples as context (if available)
#             context_examples = previous_examples[-5:]
#             examples_str = "\n".join(
#                 [
#                     f'{{"text": "{ex["text"]}", "label": "{ex["label"]}", "phenomena": {ex.get("phenomena", [])}}}'
#                     for ex in context_examples
#                 ]
#             )
#             return f"{prompt_template}\n\nReference examples:\n{examples_str}\n\nNow generate new examples."
#         else:
#             return prompt_template

#     all_examples = []

#     # Loop to generate data in batches
#     for i in range(0, num_examples, batch_size):
#         current_batch_size = min(batch_size, num_examples - i)
#         prompt = build_prompt(all_examples)
#         # Add the desired number of examples to generate in this batch
#         prompt_batch = (
#             f"{prompt}\n\nGenerate {current_batch_size} examples in JSON format."
#         )
#         print(f"Generating batch {i // batch_size + 1} with prompt:\n{prompt_batch}\n")

#         # If system_prompt is provided, use the chat template
#         if system_prompt:
#             messages = []
#             messages.append({"role": "system", "content": system_prompt})
#             messages.append({"role": "user", "content": prompt_batch})
#             text = tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#             inputs = tokenizer(text, return_tensors="pt").to(model.device)
#         else:
#             inputs = tokenizer(prompt_batch, return_tensors="pt").to(model.device)

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#         )
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Extract JSON array from the generated text
#         try:
#             batch_examples = response2json(generated_text)
#             all_examples.extend(batch_examples)
#             print(
#                 f"Batch {i // batch_size + 1} generated with {len(batch_examples)} examples."
#             )
#         except Exception as e:
#             print(f"Failed to parse batch {i // batch_size + 1}: {e}")

#     return all_examples


def generate_synthetic_data(
    prompt,
    num_examples,
    model,
    tokenizer,
    max_new_tokens=8192,
    system_prompt=None,
):
    """
    generate synthetic data in batches without rolling context

    - prompt: the prompt to generate data
    - num_examples: number of examples to generate
    - model: the model to use for generation
    - tokenizer: the tokenizer to use for generation
    - max_new_tokens: the maximum number of tokens to generate
    - system_prompt: the system prompt to use for generation

    Returns:
        List[dict]: A list of generated examples.
        int: number of reruns of the prompt taken to generate num_examples examples
    """

    all_examples = []

    # Loop to generate data in batches
    count = 0
    while len(all_examples) < num_examples:
        count += 1
        print(f"Generation number {count}")
        generated_text, _ = get_response(
            prompt,
            model,
            tokenizer,
            max_new_tokens,
            system_prompt,
            print_output=False,  # don't print
            seed=None,  # don't set seed at each iteration
        )

        match = re.search(r"```json\n(.*?)\n```", generated_text, re.DOTALL)

        if match:
            generated_text = match.group(1)  # Extract the JSON content

        try:
            batch_examples = json.loads(generated_text)  # Parse JSON string
            all_examples.extend(batch_examples)  # append to all_examples
            print(
                f"At generation {count} we have {len(all_examples)} examples out of {num_examples}"
            )
        except Exception as e:
            print(f"Failed to parse generation {count}: {e}")

        clear_cuda_cache()

    all_examples = all_examples[:num_examples]  # truncate to num_examples

    return all_examples, count


def main_generate_dataset(config):

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    config["max_new_tokens"] = config.get("max_new_tokens", 8192)
    config["system_prompt"] = config.get("system_prompt", None)

    start_time = time.time()

    # Generate data using batch generation
    # data = generate_synthetic_data_rolling(
    #     prompt_template=config["prompt"],
    #     num_examples=config["num_examples"],
    #     model=config["model"],
    #     tokenizer=config["tokenizer"],
    #     batch_size=config.get("batch_size", 50),
    #     max_new_tokens=config.get("max_new_tokens", 8192),
    # )
    data, num_runs = generate_synthetic_data(
        prompt=config["prompt"],
        num_examples=config["num_examples"],
        model=config["model"],
        tokenizer=config["tokenizer"],
        max_new_tokens=config["max_new_tokens"],
        system_prompt=config["system_prompt"],
    )
    total_time = round(time.time() - start_time, 2)

    # Log the generation details
    log_details = {
        "timestamp": datetime.now().isoformat(),
        "model": config["model"].name_or_path,
        "model.generation_config": config["model"].generation_config.to_diff_dict(),
        "model_BitsAndBytesConfig": (
            config["model"].config.quantization_config.to_diff_dict()
            if hasattr(model.config, "quantization_config")
            else None
        ),
        "generation_method": config["generation_method"],
        "prompt": config["prompt"],
        "system_prompt": config["system_prompt"],
        "time_taken_seconds": total_time,
        "num_examples_generated": len(data),
        "json_output_file": config["json_output_file"],
        "num_runs": num_runs,  # number of reruns of the prompt taken to generate num_examples examples
        "seed": seed,
    }
    log_generation(log_details, config["log_file"])

    # Prepare metadata and dataset to save as JSON
    dataset_metadata = {
        "metadata": log_details,  # metadata is saved also in the dataset
        "generated_examples": data,
    }
    save_dataset_json(dataset_metadata, config["json_output_file"])
    clear_cuda_cache(config["model"])


#############################################
# EXAMPLE USAGE
#############################################
if __name__ == "__main__":
    import os
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)
    # from src._utils._generate_dataset import main

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,  # load in 4-bit quantization
        # if I want to add other model parameters, I can add them here
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    prompt = "example prompt"

    # EXAMPLE CONFIG DICT
    config = {
        "model": model,  # THE ACTUAL MODEL OBJECT
        "tokenizer": tokenizer,  # THE ACTUAL TOKENIZER OBJECT
        "generation_method": "targeted",
        "prompt": prompt,
        "system_prompt": None,
        "num_examples": 500,
        "max_new_tokens": 8192,  # per generation call (not total)
        "seed": 42,
        "json_output_file": "synthetic_data/datasets/example.json",
        "log_file": "src/semevalirony/example_log.json",
    }
    main_generate_dataset(config)
