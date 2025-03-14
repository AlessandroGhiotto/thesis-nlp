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

sns.set(style="darkgrid")
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

    sns.histplot(filtered_df["text_length"], bins=30, kde=False, ax=axes[2])
    axes[2].set_title("Text Length Distribution")
    axes[2].set_xlabel("Number of words")
    axes[2].set_ylabel("Frequency")

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
