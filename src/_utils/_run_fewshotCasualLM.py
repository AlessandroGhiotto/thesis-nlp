import json
import os
import time
from datetime import datetime
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from src._utils._helpers import set_seed


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
    print(f"📝 Log saved successfully to: {log_file}")


def construct_fewshot_prompt(selected_examples, instruction):
    """Constructs the few-shot learning prompt"""

    def interleave_fewshot_examples(df):
        """separate examples by label and interleave them. so are presented in a balanced way"""
        labels = df["label"].unique()
        num_examples_per_label = len(df) // len(labels)
        label_to_examples = {
            label: df[df["label"] == label]
            .sample(num_examples_per_label, random_state=42)
            .values.tolist()
            for label in labels
        }
        interleaved = []
        for i in range(num_examples_per_label):
            for label in labels:
                if label_to_examples[label]:
                    interleaved.append(
                        label_to_examples[label].pop(0)
                    )  # Take one example per round
        return pd.DataFrame(interleaved, columns=df.columns)

    labels = selected_examples["label"].unique().tolist()
    selected_examples = interleave_fewshot_examples(selected_examples)
    # attach options to instruction
    options = ", ".join(labels)
    instruction += f". The options are - {options}.\n"
    # create list of elements [text: <text>. Category: <category>]
    elements = [
        f'Text: {row["text"]}. Category: {row["label"]}'
        for _, row in selected_examples.iterrows()
    ]
    # attach elements to instruction
    fewshot_prompt = instruction + "\n".join(elements)
    return fewshot_prompt


def run_fewshot_task(
    fewshot_prompt, test_df, model, tokenizer, labels, system_prompt=None
):
    """Run few-shot prompting task"""
    list_of_lengths = (lambda x: [len(i) for i in x])(labels)
    predictions = []
    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing examples"
    ):
        text = row["text"]
        # append the text to be predicted to the few-shot prompt
        prompt_text = fewshot_prompt + f"\nText: {text}. Category:"

        ## generate output
        # use chat template with system prompt if provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        # use prompt text directly
        else:
            model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # messages = []
        # if system_prompt:
        #     messages.append({"role": "system", "content": system_prompt})

        # messages.append({"role": "user", "content": fewshot_prompt})
        # messages.append({"role": "assistant", "content": f"Text: {text}. Category:"})
        # input_text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     continue_final_message=True,  # CONTINUE THE LAST MESSAGE
        # )
        # model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max(
                list_of_lengths
            ),  # TRY TO INCREASE THIS IF DON'T GET OUTPUT
        )
        output = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            output_scores=True,
        )[0]
        # extract the predicted label
        predicted_label = output.split("Category:")[-1].strip()
        # EXAMPLE predicted_label: Business\nText: NEW DELHI
        predicted_label = predicted_label.split("\n")[0].strip()
        predictions.append(predicted_label)
    res_df = test_df.copy()
    res_df.loc[:, "predicted_label"] = predictions
    return res_df


def compute_metrics(df):
    # Single-label multi-class classification
    y_pred = df["label"].tolist()
    y_true = df["predicted_label"].tolist()

    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main_fewshot_classification(config):
    """
    Main function to run few-shot classification task.

    1. construct the few-shot prompt
    2. run the few-shot task on the test data
    3. store results
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    print("\n🚀 Starting Few-Shot Classification")
    print(f"📊 Dataset         : {config.get('dataset_name', 'Not Specified')}")
    print(f"📚 Experiment      : {config.get('experiment_name', 'Not Specified')}")
    print(f"🤖 Model           : {config['model'].name_or_path}")
    print(f"💾 Output File     : {config['output_file']}")
    print(f"🎯 Seed            : {config.get('seed', 'Not Set')}\n")

    config["label"] = config["fewshot_df"]["label"].unique().tolist()
    system_prompt = config.get("system_prompt", None)

    t0 = time.time()
    # CONSTRUCT FEWSHOT PROMPT
    # few-shot df should contains n examples per label with columns ["text", "label"]
    fewshot_prompt = construct_fewshot_prompt(
        config["fewshot_df"], config["instruction"]
    )

    # RUN FEWSHOT TASK
    predictions_df = run_fewshot_task(
        fewshot_prompt,
        config["test_df"],
        config["model"],
        config["tokenizer"],
        config["label"],
        system_prompt,
    )

    # COMPUTE METRICS
    eval_metrics = compute_metrics(predictions_df)

    eval_time = time.time() - t0

    # LOGGING
    log_details = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": config["dataset_name"],
        "experiment_name": config["experiment_name"],
        "model": config["model"].name_or_path,
        "model.generation_config": config["model"].generation_config.to_diff_dict(),
        "model_BitsAndBytesConfig": (
            config["model"].config.quantization_config.to_diff_dict()
            if hasattr(config["model"].config, "quantization_config")
            else None
        ),
        "num_fewshot_examples": len(config["test_df"]),
        "label": config["label"],
        "instruction": config["instruction"],
        "system_prompt": system_prompt,
        "eval_time": eval_time,
        "log_file": config["log_file"],
        "output_file": config["output_file"],
        "seed": seed,
        "fewshot_df": config["fewshot_df"].to_dict(orient="records"),
        "metrics_test": eval_metrics,
    }
    log_generation(log_details, config["log_file"])

    predictions_df.to_csv(config["output_file"], index=False)
    print(f"💾 Predictions saved to: {config['output_file']}")

    return log_details


#############################################
# EXAMPLE USAGE
#############################################
if __name__ == "__main__":
    from transformers import LlamaTokenizer, LlamaForCausalLM

    model_name = "meta-llama/Llama-2-7b-hf"
    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Example DataFrames
    selected_examples = pd.DataFrame(
        {
            "text": ["The Earth orbits the Sun", "The stock market fluctuates"],
            "label": ["science/tech", "business"],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": [
                "Quantum mechanics is fascinating",
                "Companies report their earnings quarterly",
            ],
            "label": ["science/tech", "business"],
        }
    )

    config = {
        "dataset_name": "dataset_name",
        "experiment_name": "fewshot_classification",
        "fewshot_df": selected_examples,
        "test_df": test_df,
        "instruction": "Classify the following texts into one of the given categories",
        "model": model,
        "tokenizer": tokenizer,
        "output_file": "predictions.csv",
        "log_file": "fewshot_log.json",
        "seed": 42,
    }

    main_fewshot_classification(config)
