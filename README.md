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

## 🏗️ DevOps & Infrastructure Architecture

This project is a fully production-ready, containerized application hosted on AWS EC2, featuring a comprehensive DevOps lifecycle:

### 1. Continuous Integration (CI) Pipeline
The project utilizes **GitHub Actions** (`.github/workflows/ci.yml`) to ensure code quality and prevent broken code from reaching production. The pipeline automatically triggers on every `push` and `pull_request` to the `main` branch, performing:
- **Environment Provisioning:** Spins up an Ubuntu runner and provisions Python 3.10.
- **Dependency Caching:** Intelligently caches `pip` packages to dramatically speed up build times.
- **Dependency Installation:** Resolves and installs all requirements from `requirements.txt`.
- **Syntax Validation:** Uses `flake8` to automatically catch syntax errors, undefined names, and critical Python formatting issues.

### 2. Containerization & Registry
- **Docker:** The application is containerized using a highly optimized `Dockerfile` based on `python:3.10-slim`.
- **Resource Optimization:** We explicitly install the `CPU-only` version of PyTorch and utilize a `.dockerignore` file to prevent caching heavy dataset files, resulting in an incredibly fast and lightweight image that will not exhaust EC2 disk space.

### 3. Kubernetes (k3s) Orchestration
The application is deployed on an AWS EC2 instance using **k3s**, a highly efficient, CNCF-certified Kubernetes distribution with embedded `etcd`:
- **Redundancy:** The `nlp-app-deployment` maintains a baseline of multiple replicas ensuring high availability.
- **Horizontal Pod Autoscaling (HPA):** Dynamically scales the application out (up to 8 replicas) or in (down to 2 replicas) depending on real-time CPU utilization (target: 75%).
- **Load Balancing:** Exposes the Streamlit application seamlessly to the public internet via Kubernetes `NodePort` and `LoadBalancer` services.

### 4. Observability & Monitoring
A complete **Prometheus & Grafana** stack is deployed within the cluster to monitor system health:
- **Custom Application Metrics:** `app.py` is instrumented with the `prometheus_client` to expose a live `/metrics` endpoint. A custom `ServiceMonitor` specifically scrapes `nlp_generation_requests_total` to track exact NLP usage in Grafana.
- **Infrastructure Metrics:** Monitors EC2 CPU, Memory, Disk Pressure, and network I/O.

### 5. Automated Alerting
Using **Alertmanager**, custom Kubernetes `Secrets`, and a specialized `PrometheusRule` ConfigMap, the cluster actively analyzes metrics and triggers Slack notifications for:
- `EC2HighCpuUsage`: Server CPU exceeds 85% for 5 minutes.
- `NlpContainerHighMemory`: Any Streamlit pod exceeds 500MB of RAM.
- `NlpPodCrashing`: Immediate alert if the Streamlit container enters a `CrashLoopBackOff` state.

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
