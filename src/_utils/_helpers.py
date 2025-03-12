import json
import re
from datetime import datetime
import time
import random
import os
import numpy as np
import torch
import gc

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_cuda_cache(model=None):
    """
    Clears CUDA memory and deletes the provided model object if given.

    Parameters:
    - model (torch.nn.Module, optional): The model to delete from memory.
    """
    if model is not None:
        model.to("cpu")
        del model

    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()


def get_response(
    prompt,
    model,
    tokenizer,
    max_new_tokens=2048,
    system_prompt=None,
    print_output=True,
    seed=42,
):
    """
    Generate a response from the model given a prompt.

    Parameters:
    - prompt (str): The prompt to generate a response to.
    - model (transformers.PreTrainedModel): The model to generate the response.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    - max_new_tokens (int): The maximum number of tokens to generate.
    - system_prompt (str): The system prompt to prepend to the user prompt.
    - print_output (bool): Whether to print the generated.
    - seed (int): The random seed to use for generation.

    Returns:
    - response (str): The generated response.
    - delta_t (float): Time taken for generation in seconds.
    """
    # Set random seed if given
    # so the order in which I execute the cells does not affect the results
    if seed:
        set_seed(seed)

    t0 = time.time()
    messages = []
    # Add system prompt only if given
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Always add user prompt
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    delta_t = round(time.time() - t0, 2)
    if print_output:
        print(f"TIME TAKEN: {delta_t}\nGENERATED RESPONSE:\n{response}")

    return response, delta_t


def response2json(response):
    """
    Convert a response (str) to a list of dictionaries using json.loads().
    """
    match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

    if match:
        response = match.group(1)  # Extract the JSON content

    try:
        synthetic_data = json.loads(response)  # Parse JSON string
        return synthetic_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None  # Handle errors gracefully


def log_synthetic_data(
    model,
    generation_method,
    prompt,
    generated_samples,
    time_taken,
    output_file="synthetic_data_log.json",
):
    """
    Logs synthetic data generation details in a JSON file.

    Parameters:
    - model (str): Name of the LLM used.
    - generation_method (str): "baseline" or "targeted", ...
    - prompt (str): The prompt used for generation.
    - generated_samples (list of dict): List of generated samples, each with "text" and "label".
    - time_taken (float): Time taken for generation in seconds.
    - output_file (str): File to store the log.
    """
    if generated_samples is None:
        print("No samples to log.")
        return None

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "generation_method": generation_method,
        "prompt": prompt,
        "time_taken_seconds": round(time_taken, 2),
        "num_examples": len(generated_samples),
        "generated_examples": generated_samples,  # Store as a list of dicts with "text" and "label"
    }

    # Append to JSON file
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    print(
        f"Logged {len(generated_samples)} examples to {output_file}. Time taken: {time_taken:.2f} seconds"
    )
