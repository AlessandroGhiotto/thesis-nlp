import json
import re
import ast
from datetime import datetime
import time


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
    - generation_method (str): "baseline" or "targeted".
    - prompt (str): The prompt used for generation.
    - generated_samples (list of dict): List of generated samples, each with "text" and "label".
    - time_taken (float): Time taken for generation in seconds.
    - output_file (str): File to store the log.
    """
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


def response2json(response):
    """
    Convert a response (str) to a list of dictionaries.
    """
    match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

    if match:
        response = match.group(1)  # remove the ```json\n...\n``` part

    synthetic_data = ast.literal_eval(
        response
    )  # evaluate the string (we get a list of dictionaries)
    return synthetic_data


def get_response(
    prompt, model, tokenizer, max_new_tokens=1024, system_prompt=None, print_output=True
):
    """
    Generate a response from the model given a prompt.

    Parameters:
    - prompt (str): The prompt to generate a response to.
    - model (transformers.PreTrainedModel): The model to generate the response.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    - max_new_tokens (int): The maximum number of tokens to generate.
    - system_prompt (str): The system prompt to prepend to the user prompt.
    - print_output (bool): Whether to print the generated

    Returns:
    - response (str): The generated response.
    - delta_t (float): Time taken for generation in seconds.
    """
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
