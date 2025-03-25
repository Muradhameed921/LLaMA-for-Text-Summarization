# Fine-Tuning LLaMA for Text Summarization

## Overview
This repository contains an implementation of fine-tuning LLaMA (Large Language Model Meta AI) for text summarization. The objective is to train LLaMA to generate extractive or abstractive summaries based on input text.

## Dataset
We use a public text summarization dataset from Kaggle:
[Text Summarization Dataset](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models/input)

## Model Architecture
LLaMA is a transformer-based model optimized for natural language generation tasks. For summarization, LLaMA can be fine-tuned to either extract key sentences (extractive summarization) or generate fluent summaries (abstractive summarization).

### Fine-Tuning Process
- **Preprocessing:**
  - Tokenization using Hugging Face's `LlamaTokenizer`.
  - Padding and truncation of text sequences.

- **Training:**
  - Model: LLaMA from Hugging Face Transformers.
  - Loss Function: Cross-Entropy Loss.
  - Optimizer: AdamW.
  - Learning Rate Scheduler: Linear decay with warm-up steps.
  - Training with teacher forcing for improved sequence generation.

- **Evaluation:**
  - Loss tracking during training.
  - Summarization performance comparison using ROUGE-N and ROUGE-L scores.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install transformers datasets torch nltk rouge-score
```

## Running the Notebook
1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-text-summarization.git
cd llama-text-summarization
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook Llama.ipynb
```
3. Follow the steps in the notebook to preprocess the data, fine-tune LLaMA, and evaluate the model.

## Results
- **Loss Analysis:**
  - Training loss is reported for each epoch.
- **Sample Summaries:**
  - Example original text and generated summaries are displayed.
- **Comparison:**
  - Summary quality is compared using ROUGE scores.

## Future Improvements
- Experiment with different fine-tuning strategies.
- Compare performance with BERT and GPT-2 for text summarization.
- Deploy the trained model as an API for real-world usage.

## Output
![Llama Text Summarization Screenshot](https://github.com/Muradhameed921/LLaMA-for-Text-Summarization/blob/main/Llama.png)

