# Text classification

### 📂 Project Structure

```bash
bai-thesis-nlp/
├── src/
│   ├── dataset1/       # notebooks, scripts and models for this dataset
│   │   ...
│   ├── datasetN/
│   ├── utils/          # utility functions and helper code
│   └── misc/           # assorted notebooks and files
├── realdata/           # real datasets
└── synthetic_data/     # generated synthetic data in JSON format
    ├── logs/           # logs of toy synthetic data
    └── datasets/       # larger synthetic datasets
```

### 📄 Synthetic Data Format

The JSON files in `synthetic_data/logs/` follow this structure:

```
{
    "timestamp": "<timestamp>",
    "model": "<model_name>",
    "generation_method": "baseline" | "targeted" | "targeted + linguistic tags",
    "prompt": "<prompt_used>",
    "time_taken_seconds": <time_taken>,
    "num_examples": <number_of_examples>,
    "generated_examples": [
        {
            "text": "<generated_text>",
            "label": "<corresponding_label>",
            "phenomena": ["<phenomenon1>", "<phenomenon2>", ...] (optional)
        }
    ]
}

```
