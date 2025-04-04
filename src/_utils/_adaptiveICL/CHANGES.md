# Changes I have made to adaptiveICL

- utils.arg_parser(), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" as model choice
- utils.arg_parser(), --dataset accept any string (no more a list of specified choices), for more flexibility
- utils.load_model_tokenizer(), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"...
- utils.load_model_tokenizer(), Quantize everytime the model
- main, commented model.to(device) since the quantized model is already in the correct device
- donwloaded modules (in addition to environmnet.yml): BitsAndBytes, ipykernel, matplotlib, seaborn, wordcloud
- classification_report(..., zero_division=0)
- minor changes in the prints
- for adding a new dataset (like agnews):
  - config.get_config(), "agnews" in dataset:...
  - utils.arg_parser(), "agnews" as --dataset choice
  - utils, csvProcessor for csv files with cols=['text', 'label']
  - utils.load_dataset(), elif "agnews" in dataset:...
  - utils.pred_batch(), elif "agnews" in dataset:...
    (used in instead of == to accept more flexibility)
