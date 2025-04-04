# Changes I have made to adaptiveICL

- utils.arg_parser(), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" as model choice
- utils.load_model_tokenizer(), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"...
- utils.load_model_tokenizer(), Quantized everytime the model
- main, commented model.to(device) since the quantized model is already in the correct device
- donwloaded modules (in addition to environmnet.yml): BitsAndBytes, ipykernel, matplotlib, seaborn, wordcloud
- classification_report(..., zero_division=0)
- minor changes in the prints
- agnews dataset:
  - config.get_config(), dataset == "agnews":...
  - utils.arg_parser(), "agnews" as --dataset choice
  - utils, csvProcessor for csv files with cols=['text', 'label']
  - utils.load_dataset(), elif dataset == "agnews":...
  - utils.pred_batch(), elif dataset == "agnews":...
