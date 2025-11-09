# NLP Project - Product Description Generator

A GPT-2 based product description generator with evaluation metrics (BLEU, ROUGE) and iterative feedback loop.

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
- Batch processing with iterative feedback
- Real-time metrics visualization
- Adjustable generation parameters
- Manual feedback scoring



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

### Key Features:
- **SmoothingFunction**: Prevents BLEU score warnings when n-grams don't match
- **Proper Tokenization**: Sets pad_token to avoid warnings
- **Text Extraction**: Separates prompt from generated text
- **Comprehensive Metrics**: Returns BLEU, ROUGE-1, ROUGE-2, and ROUGE-L scores
- **Error Handling**: Graceful error handling throughout

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
