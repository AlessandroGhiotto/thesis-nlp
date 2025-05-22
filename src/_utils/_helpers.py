import json
import re
from datetime import datetime
import time
import random
import os
import numpy as np
import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set_style("darkgrid")
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
    messages=None,
    print_output=True,
    seed=42,
    apply_chat_template=True,
):
    """
    Generate a response from a model given a prompt or message list.

    Parameters:
    - prompt (str): User prompt to generate a response to.
    - model (transformers.PreTrainedModel): The model to generate the response.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    - max_new_tokens (int): Maximum number of tokens to generate.
    - system_prompt (str): Optional system prompt prepended to the user prompt.
    - messages (list): Optional list of chat messages (dicts with 'role' and 'content').
    - print_output (bool): Whether to print the generated response.
    - seed (int): Random seed for deterministic generation.
    - apply_chat_template (bool): Whether to apply the chat template (for chat models).

    Returns:
    - response (str): The generated text response.
    - delta_t (float): Time taken for generation in seconds.
    """

    if seed is not None:
        set_seed(seed)

    t0 = time.time()

    if apply_chat_template:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        elif system_prompt:
            # insert system prompt only if not already in messages
            if not any(m["role"] == "system" for m in messages):
                messages = [{"role": "system", "content": system_prompt}] + messages

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).strip()
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    else:
        # plain prompt (no chat template)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    # Remove the prompt part from output
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    delta_t = round(time.time() - t0, 2)

    if print_output:
        print(f"TIME TAKEN: {delta_t:.2f} seconds\nGENERATED RESPONSE:\n{response}")

    return response, delta_t


def response2json(response):
    """
    Convert a response (str) to a list of dictionaries using json.loads().
    """
    json_str = extract_json_from_text(response)

    if not json_str:
        print("No JSON-like content found in the response.")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def extract_json_from_text(text):
    """
    Extracts JSON content from a string using multiple fallbacks.

    Tries:
    1. ```json\n...\n```
    2. First object or array-looking structure: {} or [{}...]

    Returns:
    - str or None: JSON string if found, else None
    """
    # Try triple-backtick json blocks first
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try finding a list of objects: [ {...}, {...} ]
    match = re.search(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Try finding a single object: { ... }
    match = re.search(r"\{\s*.*?\s*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def fix_missing_commas(json_str):
    """
    Attempts to fix missing commas between string fields in JSON objects.
    works just in simple cases (such as ours)
    1. Between object fields (e.g., "a": 1 "b": 2 → "a": 1, "b": 2)
    2. Between list elements (e.g., {...}{...} → {...}, {...})
    """
    # Fix missing commas between fields inside an object
    json_str = re.sub(
        r'(":[^"]*?")\s*(")',  # Matches `": value" "next_field"` without comma
        r"\1, \2",  # Inserts comma: `": value", "next_field"`
        json_str,
    )

    # Fix missing commas between objects in a list (e.g., }{ → }, {)
    json_str = re.sub(
        r"(\})\s*(\{)", r"\1, \2", json_str  # Matches }{  # Replace with },{
    )

    return json_str


def fix_missing_commas(json_str):
    """
    Fixes:
    1. Missing commas between fields in objects.
    2. Missing commas between objects in a list.
    3. Trailing commas before closing brackets.
    """
    # Fix missing commas between fields in objects (e.g., "val" "key" → "val", "key")
    json_str = re.sub(
        r'(":[^"]*?")\s*(")',  # Match string value followed by another field name
        r"\1, \2",
        json_str,
    )

    # Fix missing commas between adjacent objects (e.g., }{ → }, {)
    json_str = re.sub(r"(\})\s*(\{)", r"\1, \2", json_str)

    # Optional: Remove trailing commas before closing array or object brackets (last element)
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    return json_str


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
    Here we have small batches of generated samples

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

    generated_samples = (
        generated_samples
        if isinstance(generated_samples, list)
        else [generated_samples]
    )

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


def get_generated_examples_df(path):
    """
    Load the generated examples from a JSON file into a DataFrame.
    JSON file should have the following structure:
    {
        "metadata": {...},
        "generated_examples": [
            {"text": "Example 1 text here", "label": "Example 1 label here"},
            {"text": "Example 2 text here", "label": "Example 2 label here"},
            ...
        ]
    }

    return:
    - df (pd.DataFrame): The DataFrame containing the generated examples.
    - metadata (dict): The metadata from the JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    df = pd.DataFrame(data["generated_examples"])
    return df, metadata


def basic_analysis(df, print_missing_values=True, print_count_statistics=True):
    """
    df should be a pandas DataFrame with the columns 'text' and 'label'

    This function will:
    - Display a WordCloud of the text corpus
    - Display a Pie Chart of the label distribution
    - Display a Histogram of the text length distribution
    - Print basic statistics about the dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- 1️ WordCloud ----
    text_corpus = " ".join(str(text) for text in df["text"])
    wordcloud = WordCloud(
        width=800, height=800, background_color="white", colormap="viridis"
    ).generate(text_corpus)
    axes[0].imshow(wordcloud, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("WordCloud of Dataset")

    # ---- 2️ Pie Chart for Label Distribution ----
    label_counts = df["label"].value_counts()

    # Reduce label size if too many labels
    if len(label_counts) >= 10:
        label_fontsize = 8
    else:
        label_fontsize = 12
    axes[1].pie(
        label_counts,
        labels=label_counts.index,
        autopct="%1.1f%%",
        colors=plt.cm.Paired.colors,
        textprops={"fontsize": label_fontsize},
    )
    axes[1].set_title("Label Distribution")

    # ---- 3️ Text Length Distribution ----
    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))  # Count words
    # Filter the dataset (just for visualization purposes)
    threshold = df["text_length"].quantile(0.995)  # Keep 99.5% of data
    filtered_df = df[df["text_length"] <= threshold]

    mean_length = df["text_length"].mean()

    sns.histplot(
        filtered_df["text_length"], bins=30, kde=False, ax=axes[2], label="Text Lengths"
    )
    axes[2].axvline(
        mean_length, color="red", linestyle="--", label=f"Mean = {mean_length:.2f}"
    )
    axes[2].set_title("Text Length Distribution")
    axes[2].set_xlabel("Number of words")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # ---- Print basic statistics ----
    print(f"Number of train samples: {len(df)}")
    print(f"Set of labels: {set(df['label'])}")
    print(f"Number of labels: {len(set(df['label']))}")
    if print_missing_values:
        print(f"Missing Values: \n{df.isnull().sum()}", sep="")
    if print_count_statistics:
        print("\nWord Count Statistics:")
        print(df["text_length"].describe().round(2))

    return None


def get_context_examples(df, num_examples_per_prompt, num_prompts):
    """
    Function used for sampling context examples for the unsupervised generation

    we sample randomly "num_examples_per_prompt" examples, and we do it "num_prompts" times
    we store them in a list of lists
    """
    context_examples = []
    for _ in range(num_prompts):
        examples = df.sample(
            n=num_examples_per_prompt,
            replace=False,
            random_state=np.random.randint(0, 1e6),
        )
        context_examples.append(examples["text"].tolist())
    return context_examples
