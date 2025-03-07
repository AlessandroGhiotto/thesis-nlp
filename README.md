# Text classification

### 📂 Project Structure

- `mycode/` – Contains the Jupyter Notebooks.
- `synthetic_data/` – Stores generated synthetic data in JSON format.

### 📄 Synthetic Data Format

The JSON files in `synthetic_data/` follow this structure:

```javascript
{
    "timestamp": <timestamp>,
    "model": <model_name>,
    "generation_method": "baseline" | "targeted",
    "prompt": "<prompt_used>",
    "time_taken_seconds": "time_taken",
    "num_examples": number_of_examples,
    "generated_examples": [
        {
            "text": <generated_text>,
            "label": <corresponding_label>
        }
    ]
}
```
