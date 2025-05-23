import json
import os
import time
from datetime import datetime
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src._utils._helpers import (
    get_response,
    set_seed,
    clear_cuda_cache,
    extract_json_from_text,
    fix_missing_commas,
)


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


def save_dataset_json(metadata, output_file):
    """Save the dataset along with metadata in a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def get_valid_examples(examples, correct_labels=None, correct_fields=None):
    """
    Remove the examples that have a non-valid label or fields.
    """

    correct_labels = set(correct_labels) if correct_labels else None
    correct_fields = set(correct_fields) if correct_fields else None

    valid_examples = []
    for ex in examples:
        # skip example if the label is not one of the real labels
        if correct_labels and ex.get("label") not in correct_labels:
            continue
        # skip example if the fields are not correct
        if correct_fields and set(ex.keys()) != correct_fields:
            continue
        valid_examples.append(ex)

    return valid_examples


def format_context_examples(context_examples, example_format):
    ### The intro "Here are some examples..." must be printed in the base prompt
    # here we have only the examples
    if example_format == "bullet":
        # when an example is a string
        context_str = "".join(f"- {ex}\n" for ex in context_examples)
    elif example_format == "json":
        # when an example is a list of dicts
        context_str = (
            "```json\n"
            + json.dumps(context_examples, indent=4, ensure_ascii=False)
            + "\n```\n"
        )
    else:
        context_str = str(context_examples)

    return context_str


def format_prompt_with_context(
    base_prompt, context_examples, example_format="bullet", postfix=None
):
    """
    Format the prompt by including context examples, with optional postfix.
    This is used when the prompt is a string.

    base_prompt + context examples + postfix
    """
    prompt_parts = []
    if base_prompt:
        prompt_parts.append(base_prompt)
    if context_examples:
        context_str = format_context_examples(
            context_examples, example_format=example_format
        )
        prompt_parts.append(context_str)
    if postfix:
        prompt_parts.append(postfix)
    return "\n".join([part for part in prompt_parts if part])


def insert_context_in_messages(
    messages,
    context_examples,
    placeholder="ADD_CONTEXT_HERE",
    example_format="bullet",
    postfix=None,
):
    """
    Replace placeholder in user messages with context and optional postfix.
    This is used when we have a list of messages.

    the context (lsit of examples) is injected in the user message
    where the placeholder is found.
    """
    context_str = ""
    if context_examples:
        context_str = format_context_examples(
            context_examples, example_format=example_format
        )
    if postfix:
        context_str += f"\n{postfix}"

    # Replace the placeholder in user messages with context
    updated_messages = []
    for msg in messages:
        new_msg = msg.copy()
        if msg["role"] == "user" and placeholder in msg["content"]:
            new_msg["content"] = new_msg["content"].replace(placeholder, context_str)
        updated_messages.append(new_msg)
    return updated_messages


def insert_label_in_messages(messages, label, placeholder="ADD_LABEL_HERE"):
    """
    Replace placeholder in user messages with label.
    This is used when we have a list of messages.
    the label is injected in the user message
    where the placeholder is found.
    """
    # Replace the placeholder in user messages with label
    updated_messages = []
    for msg in messages:
        new_msg = msg.copy()
        if msg["role"] == "user" and placeholder in msg["content"]:
            new_msg["content"] = new_msg["content"].replace(placeholder, label)
        updated_messages.append(new_msg)
    return updated_messages


def generate_synthetic_data(
    prompt,
    num_examples,
    model,
    tokenizer,
    max_new_tokens=8192,
    system_prompt=None,
    correct_labels=None,
    correct_fields=["text", "label"],
    apply_chat_template=True,
    add_random_label=False,
    label_placeholder="ADD_LABEL_HERE",
):
    """
    generate synthetic data in batches without rolling context

    - prompt: the prompt to generate data
    - num_examples: number of examples to generate
    - model: the model to use for generation
    - tokenizer: the tokenizer to use for generation
    - max_new_tokens: the maximum number of tokens to generate
    - system_prompt: the system prompt to use for generation
    - correct_labels: the correct labels to use for generation
    - correct_fields: the correct fields to be obtained in the generation
    - apply_chat_template: whether to apply the chat template to the prompt
    - add_random_label: whether to append a random label to the prompt messages (for chat template)

    Returns:
        List[dict]: A list of generated examples.
        int: number of reruns of the prompt taken to generate num_examples examples
    """

    correct_labels = set(correct_labels) if correct_labels else None
    correct_labels_list = list(correct_labels) if correct_labels else None
    correct_fields = set(correct_fields) if correct_fields else None
    all_samples = []

    # Determine if prompt is a list of messages
    # the messages are intended to be used in the chat format
    messages = prompt if isinstance(prompt, list) else None
    user_prompt = None if messages else prompt

    # Loop to generate data in batches
    with tqdm(total=num_examples, desc="Generating Examples", unit="ex") as pbar:
        run_number = 1
        while len(all_samples) < num_examples:
            run_number += 1

            # Add a random label if requested
            if add_random_label and messages:
                random_label = random.choice(correct_labels_list)
                messages_with_label = insert_label_in_messages(
                    messages, random_label, placeholder=label_placeholder
                )
            else:
                messages_with_label = messages

            generated_text, _ = get_response(
                prompt=user_prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                messages=messages_with_label,
                print_output=False,
                seed=None,
                apply_chat_template=apply_chat_template,
            )
            json_str = extract_json_from_text(generated_text)

            if json_str is None:
                tqdm.write(f"❌ No valid JSON found in generation {run_number}")
                continue
            try:
                sample = json.loads(json_str)
            except Exception as e:
                # Try fixing common formatting issues
                fixed_json_str = fix_missing_commas(json_str)
                try:
                    sample = json.loads(fixed_json_str)
                except Exception as e2:
                    tqdm.write(f"❌ Failed to parse generation {run_number}: {e2}")
                    continue

            sample = sample if isinstance(sample, list) else [sample]
            sample = get_valid_examples(sample, correct_labels, correct_fields)
            all_samples.extend(sample)

            # Ensure we don't exceed the required number of examples
            current_count = min(len(all_samples), num_examples)
            pbar.n = current_count
            pbar.set_postfix(
                run=f"{run_number}", examples=f"{current_count}/{num_examples}"
            )
            pbar.update(0)  # Refresh the bar without incrementing

            clear_cuda_cache()

    all_samples = all_samples[:num_examples]  # truncate to num_examples

    return all_samples, run_number - 1


def generate_synthetic_data_with_context(
    prompt,
    num_examples,
    model,
    tokenizer,
    context_examples=None,
    max_new_tokens=8192,
    system_prompt=None,
    correct_labels=None,
    correct_fields=["text", "label"],
    postfix=None,
    apply_chat_template=True,
    add_random_label=False,
    example_format="bullet",
):
    """
    Generate synthetic data one at a time, using context examples in the prompt.
    - context_examples: list of lists of dicts, each sublist is the context for one generation.
    - postfix: string to append after the prompt.
    """
    correct_labels = set(correct_labels) if correct_labels else None
    correct_labels_list = list(correct_labels) if correct_labels else None
    correct_fields = set(correct_fields) if correct_fields else None
    all_samples = []

    if not context_examples:
        context_examples = [None] * num_examples
    elif len(context_examples) < num_examples:
        context_examples = context_examples + [None] * (
            num_examples - len(context_examples)
        )

    idx = 0
    with tqdm(
        total=num_examples, desc="Generating Examples (context)", unit="ex"
    ) as pbar:
        while len(all_samples) < num_examples:
            if idx >= len(context_examples):
                tqdm.write("❌ Used all the examples.")
                break

            this_context = context_examples[idx]

            # Build prompt with context and label
            if isinstance(prompt, list):  # messages format
                messages = prompt
                messages = insert_context_in_messages(
                    messages,
                    this_context,
                    example_format=example_format,
                    placeholder="ADD_CONTEXT_HERE",
                    postfix=postfix,
                )
                text_prompt = None
                if add_random_label and correct_labels_list:
                    random_label = random.choice(correct_labels_list)
                    messages = insert_label_in_messages(
                        messages,
                        label=random_label,
                        placeholder="ADD_LABEL_HERE",
                    )

            else:  # string format
                messages = None
                text_prompt = format_prompt_with_context(
                    prompt, this_context, postfix=postfix, example_format=example_format
                )
                if add_random_label and correct_labels_list:
                    random_label = random.choice(correct_labels_list)
                    text_prompt += f"\nlabel: {random_label}"

            generated_text, _ = get_response(
                prompt=text_prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                messages=messages,
                print_output=False,
                seed=None,
                apply_chat_template=apply_chat_template,
            )

            json_str = extract_json_from_text(generated_text)

            if json_str is None:
                tqdm.write(f"❌ No valid JSON found at run {idx}")
                idx += 1
                continue
            try:
                sample = json.loads(json_str)
            except Exception as e:
                # Try fixing common formatting issues
                fixed_json_str = fix_missing_commas(json_str)
                try:
                    sample = json.loads(fixed_json_str)
                except Exception as e2:
                    tqdm.write(f"❌ Failed to parse generation {idx}: {e2}")
                    idx += 1
                    continue

            sample = sample if isinstance(sample, list) else [sample]
            sample = get_valid_examples(sample, correct_labels, correct_fields)
            all_samples.extend(sample)
            if sample:
                out_sample = dict(sample[0])
                out_sample["context_examples"] = this_context
                if add_random_label:
                    out_sample["requested_label"] = random_label
                all_samples.append(out_sample)
            else:
                tqdm.write(f"❌ Invalid example at run {idx}")

            pbar.n = len(all_samples)
            pbar.set_postfix(
                run=f"{idx}", examples=f"{len(all_samples)}/{num_examples}"
            )
            pbar.update(0)
            clear_cuda_cache()
            idx += 1

    return all_samples, idx


def main_generate_dataset(config):
    """ "
    Generate synthetic data using the specified configuration.

    config: dict containing the following keys:
        - model (transformers.PreTrainedModel): The model to use for generation.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        - generation_method (str): The method to use for generation.
        - prompt (str): The prompt to use for generation.
        - system_prompt (str): The system prompt to prepend to the user prompt.
        - num_examples (int): The number of examples to generate.
        - max_new_tokens (int): The maximum number of tokens to generate.
        - seed (int): The random seed to use for generation.
        - json_output_file (str): The path to save the generated dataset as JSON.
        - log_file (str): The path to save the generation log as JSON.
        - context_examples (list): Context examples to include in the prompt.
        - prompt_postfix (str): Postfix to append to the prompt.
    """
    if config.get("verbose", False):
        print("\n🚀 Starting Synthetic Dataset Generation")
        print(f"📊 Dataset              : {config.get('dataset', 'Not Specified')}")
        print(f"📚 Generation method    : {config['generation_method']}")
        print(f"🤖 Model                : {config['model'].name_or_path}")
        # print(f"📝 Prompt               : {config['prompt'][:100]}{'...' if len(config['prompt']) > 100 else ''}")
        print(f"🔢 Examples to Generate : {config['num_examples']}")
        print(f"💾 Output File          : {config['json_output_file']}")
        print(f"🕹️ Max New Tokens       : {config['max_new_tokens']}")
        print(f"🎯 Seed                 : {config.get('seed', 'Not Set')}\n")

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    config["max_new_tokens"] = config.get("max_new_tokens", 2048)
    config["system_prompt"] = config.get("system_prompt", None)
    config["correct_labels"] = config.get("correct_labels")
    config["correct_fields"] = config.get("correct_fields", ["text", "label"])
    config["apply_chat_template"] = config.get("apply_chat_template", True)
    config["prompt_postfix"] = config.get("prompt_postfix", None)
    config["add_random_label"] = config.get("add_random_label", False)
    context_examples = config.get("context_examples", None)

    start_time = time.time()

    if context_examples:
        data, num_runs = generate_synthetic_data_with_context(
            prompt=config["prompt"],
            num_examples=config["num_examples"],
            model=config["model"],
            tokenizer=config["tokenizer"],
            context_examples=context_examples,
            max_new_tokens=config["max_new_tokens"],
            system_prompt=config["system_prompt"],
            correct_labels=config["correct_labels"],
            correct_fields=config["correct_fields"],
            postfix=config["prompt_postfix"],
            apply_chat_template=config["apply_chat_template"],
            add_random_label=config["add_random_label"],
        )
    else:
        data, num_runs = generate_synthetic_data(
            prompt=config["prompt"],
            num_examples=config["num_examples"],
            model=config["model"],
            tokenizer=config["tokenizer"],
            max_new_tokens=config["max_new_tokens"],
            system_prompt=config["system_prompt"],
            correct_labels=config["correct_labels"],
            correct_fields=config["correct_fields"],
            apply_chat_template=config["apply_chat_template"],
            add_random_label=config["add_random_label"],
        )
    total_time = round(time.time() - start_time, 2)

    # Log the generation details
    log_details = {
        "timestamp": datetime.now().isoformat(),
        "dataset": config.get("dataset", "Not Specified"),
        "generation_method": config["generation_method"],
        "num_examples_generated": len(data),
        "model": config["model"].name_or_path,
        "model.generation_config": config["model"].generation_config.to_diff_dict(),
        "model_BitsAndBytesConfig": (
            config["model"].config.quantization_config.to_diff_dict()
            if hasattr(config["model"].config, "quantization_config")
            else None
        ),
        "prompt": config["prompt"],
        "prompt_postfix": config["prompt_postfix"],
        "system_prompt": config["system_prompt"],
        "apply_chat_template": config["apply_chat_template"],
        "time_taken_seconds": total_time,
        "json_output_file": config["json_output_file"],
        "num_runs": num_runs,  # number of reruns of the prompt taken to generate num_examples examples
        "seed": seed,
        "correct_labels": config["correct_labels"],
        "correct_fields": config["correct_fields"],
        "add_random_label": config["add_random_label"],
    }
    log_generation(log_details, config["log_file"])

    # Prepare metadata and dataset to save as JSON
    dataset_metadata = {
        "metadata": log_details,  # metadata is saved also in the dataset
        "generated_examples": data,
    }
    save_dataset_json(dataset_metadata, config["json_output_file"])

    print(f"⏱️ Time taken: {total_time} seconds.")
    if config.get("verbose", False):
        print(f"📝 Log saved successfully to: {config['log_file']}")
        print(f"💾 Dataset with metadata saved to: {config['json_output_file']}")


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
        "dataset": "dataset_name",
        "model": model,  # THE ACTUAL MODEL OBJECT
        "tokenizer": tokenizer,  # THE ACTUAL TOKENIZER OBJECT
        "generation_method": "targeted",
        "prompt": prompt,
        "system_prompt": None,
        "num_examples": 500,
        "max_new_tokens": 8192,  # per generation call (not total)
        "seed": 42,
        "json_output_file": "synthetic_data/datasets/DeepSeek-R1-Distill-Qwen-1.5B/example.json",
        "log_file": "src/semevalirony/example_log.json",
        "correct_labels": ["positive", "negative"],
        "correct_fields": ["text", "label"],
        "context_examples": [
            ["example1", "example2"],
            ["example3", "example4"],
        ],  # list of lists like object
        "prompt_postfix": "Postfix text",
    }
    main_generate_dataset(config)
