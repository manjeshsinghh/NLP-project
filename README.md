# NLP Project - Product Description Generator

An instruction-tuned FLAN-T5 based product description generator with evaluation metrics (BLEU, ROUGE) and flexible dataset input.

## 🌐 Live App

**Access the application:** [nlpmodel.streamlit.app](https://nlpmodel.streamlit.app)

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/manjeshsinghh/NLP-project.git
cd NLP-project
```

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
```



**Features:**
- Single product description generation
- Dataset product selection with search
- Small batch description generation
- CSV/ZIP upload support
- Flexible column mapping for different ecommerce datasets
- Real-time metrics visualization
- Adjustable generation parameters

## 🏗️ DevOps & Infrastructure

This project has been upgraded from a simple Python script to a fully production-ready, containerized application hosted on AWS EC2:

- **Docker Containerization**: Optimized `Dockerfile` using `python:3.10-slim` with CPU-only PyTorch to minimize image size and save disk space.
- **Kubernetes (k3s)**: Lightweight Kubernetes cluster managing the application lifecycle.
- **Automated CI/CD**: GitHub Actions pipeline (`.github/workflows/ci.yml`) automatically checks syntax and installs dependencies on every push.
- **Horizontal Pod Autoscaling (HPA)**: Kubernetes automatically scales the Streamlit pods (between 2 and 8 replicas) based on CPU usage.
- **Prometheus & Grafana Monitoring**: Full observability stack tracking EC2 metrics, pod health, and custom Streamlit metrics (like `nlp_generation_requests_total`).
- **Automated Alerts**: Custom Alertmanager setup using Kubernetes `Secrets` and `PrometheusRule` ConfigMaps to send Slack notifications for high CPU, memory leaks, or CrashLooping pods.

## Project Structure

```
NLP-project/
├── app.py                 # Streamlit web application
├── nlp_model.py          # Core NLP model and functions
├── NLP_project.ipynb     # Jupyter notebook
├── requirements.txt      # Python dependencies
├── amazon.csv.zip        # Dataset
└── README.md            # This file
```

## Code Improvements

### Fixed Issues:
1. ✅ **BLEU Score Warnings**: Added SmoothingFunction to handle zero n-gram overlaps
2. ✅ **Tokenizer Warnings**: Properly set pad_token and attention_mask
3. ✅ **Text Extraction**: Extract only newly generated text (excluding prompt)
4. ✅ **Error Handling**: Added try-except blocks and validation
5. ✅ **Input Handling**: Removed interactive input() for notebook compatibility
6. ✅ **Dataset Loading**: Flexible path handling for different environments
7. **Flexible Product Selection**: Dataset mode can use any matching product row, not only the first few rows
8. **General Dataset Support**: Added column mapping so datasets with names like `title`, `description`, or `category` can be used
9. **Model Quality**: Replaced base GPT-2 with instruction-tuned FLAN-T5 for more controlled product generation

### Key Features:
- **SmoothingFunction**: Prevents BLEU score warnings when n-grams don't match
- **Proper Tokenization**: Sets pad_token to avoid warnings
- **Text Extraction**: Separates prompt from generated text
- **Comprehensive Metrics**: Returns BLEU, ROUGE-1, ROUGE-2, and ROUGE-L scores
- **Error Handling**: Graceful error handling throughout
- **Reusable Prompt Builder**: Creates stronger prompts from product name, details, category, and optional extra columns
- **Quality Guardrails**: Falls back to product-focused copy if the model repeats labels, writes too little, or drifts to the wrong product

## Parameters

### Generation Parameters:
- `max_new_tokens`: Maximum number of tokens to generate (default: 150)
- `temperature`: Sampling temperature (default: 0.7)
- `top_k`: Top-k sampling (default: 50)
- `top_p`: Top-p (nucleus) sampling (default: 0.95)

### Evaluation Metrics:
- **BLEU Score**: N-gram precision between generated and reference text
- **ROUGE-1**: Unigram overlap (precision, recall, F1)
- **ROUGE-2**: Bigram overlap (precision, recall, F1)
- **ROUGE-L**: Longest common subsequence based metrics
- **Combined Score**: Weighted combination of BLEU and ROUGE-L

## Requirements

- Python 3.8+
- transformers >= 4.30.0
- torch >= 2.0.0
- nltk >= 3.8
- rouge >= 1.0.1
- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0


## License

This project is open source and available under the MIT License.

## Author

Manjesh Singh
