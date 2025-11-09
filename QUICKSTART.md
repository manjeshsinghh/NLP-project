# Quick Start Guide

## Running the Streamlit App

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**:
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

## Using the App

### Single Product Generation
1. Go to the "📝 Single Product" tab
2. Enter product name and description
3. Optionally enter reference text for evaluation
4. Adjust generation parameters in the sidebar (temperature, top-k, etc.)
5. Click "Generate Description"
6. View the generated text and evaluation metrics

### Batch Processing
1. Go to the "📊 Batch Processing" tab
2. Click "Load Dataset" in the sidebar
3. Select a product from the dropdown
4. Click "Process Product"
5. Rate each iteration using the slider (1-10)
6. View metrics and combined rewards

## Troubleshooting

### Model Loading Issues
- First run will download GPT-2 model (~500MB)
- Ensure you have internet connection
- Check available disk space

### Dataset Loading Issues
- Make sure `amazon.csv.zip` is in the same directory as `app.py`
- Check file path in the sidebar
- Verify the CSV file is not corrupted

### Memory Issues
- Reduce `max_new_tokens` in sidebar
- Process fewer products at a time
- Close other applications to free up memory

### CUDA/GPU Issues
- The app automatically uses CPU if GPU is not available
- GPU acceleration requires CUDA-compatible PyTorch installation
- CPU mode is slower but works fine for testing

## Example Usage

### Generate a Product Description
```
Product Name: Wireless Bluetooth Headphones
Description: Noise-cancelling headphones with 30-hour battery life
Reference Text: Great sound quality, comfortable fit, long battery life
```

### Expected Output
- Generated product description
- BLEU score (0.0 - 1.0)
- ROUGE-1, ROUGE-2, ROUGE-L scores
- Combined metric score

## Tips

1. **Temperature**: Lower (0.1-0.5) for more focused text, higher (0.8-1.5) for more creative text
2. **Top-K**: Higher values (50-100) for more diverse outputs
3. **Top-P**: Lower values (0.5-0.7) for more focused outputs
4. **Max Tokens**: Adjust based on desired description length

## Support

For issues or questions, please check:
- README.md for detailed documentation
- Code comments in `nlp_model.py` and `app.py`
- Jupyter notebook `NLP_project.ipynb` for examples

