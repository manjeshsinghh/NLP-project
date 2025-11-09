"""
NLP Model for Product Description Generation using GPT-2
Fixed version with proper error handling and improved metrics
"""
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ProductDescriptionGenerator:
    def __init__(self, model_name="gpt2", device=None):
        """Initialize the GPT-2 model and tokenizer"""
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name} on device: {self.device}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad_token to eos_token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize ROUGE scorer
        self.rouge = Rouge()
        
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
        
        print("Model loaded successfully!")
    
    def generate_text(self, prompt, max_new_tokens=150, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generate text using GPT-2 model
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated text (only the new part, excluding the prompt)
        """
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Ensure the length of input doesn't exceed GPT-2's maximum
            max_input_length = 1024
            input_length = inputs.shape[1]
            
            if input_length > max_input_length:
                inputs = inputs[:, -max_input_length:]
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones(inputs.shape, device=self.device)
                )
            
            # Decode the full output
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated part (remove the prompt)
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                generated_text = full_text.strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""
    
    def reward_function(self, generated_text, reference_text):
        """
        Calculate reward using BLEU and ROUGE scores
        
        Args:
            generated_text: Generated text
            reference_text: Reference text for comparison
            
        Returns:
            Dictionary with individual and combined scores
        """
        try:
            # Calculate BLEU score with smoothing
            reference_tokens = reference_text.split()
            generated_tokens = generated_text.split()
            
            if len(reference_tokens) == 0 or len(generated_tokens) == 0:
                bleu_score = 0.0
            else:
                bleu_score = sentence_bleu(
                    [reference_tokens],
                    generated_tokens,
                    smoothing_function=self.smoothing
                )
            
            # Calculate ROUGE scores
            try:
                rouge_scores = self.rouge.get_scores(generated_text, reference_text)
                rouge_l_score = rouge_scores[0]['rouge-l']['f']
                rouge_1_score = rouge_scores[0]['rouge-1']['f']
                rouge_2_score = rouge_scores[0]['rouge-2']['f']
            except:
                # If ROUGE calculation fails, set scores to 0
                rouge_l_score = 0.0
                rouge_1_score = 0.0
                rouge_2_score = 0.0
            
            # Combine BLEU and ROUGE-L scores
            combined_score = 0.5 * bleu_score + 0.5 * rouge_l_score
            
            return {
                'bleu': bleu_score,
                'rouge_1': rouge_1_score,
                'rouge_2': rouge_2_score,
                'rouge_l': rouge_l_score,
                'combined': combined_score
            }
            
        except Exception as e:
            print(f"Error calculating reward: {str(e)}")
            return {
                'bleu': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'combined': 0.0
            }
    
    def iterative_feedback(self, prompt, reference_text, iterations=5, manual_scores=None):
        """
        Iterative feedback loop for text generation
        
        Args:
            prompt: Input prompt
            reference_text: Reference text for evaluation
            iterations: Number of iterations
            manual_scores: List of manual scores (for Streamlit integration)
            
        Returns:
            List of results for each iteration
        """
        results = []
        
        for i in range(iterations):
            # Generate text
            generated_text = self.generate_text(prompt)
            
            # Calculate automatic metrics
            reward_scores = self.reward_function(generated_text, reference_text)
            
            # Get manual feedback (if provided)
            manual_score = manual_scores[i] if manual_scores and i < len(manual_scores) else None
            
            # Combine rewards
            if manual_score is not None:
                combined_reward = (0.7 * manual_score / 10) + (0.3 * reward_scores['combined'])
            else:
                combined_reward = reward_scores['combined']
            
            result = {
                'iteration': i + 1,
                'generated_text': generated_text,
                'reward_scores': reward_scores,
                'manual_score': manual_score,
                'combined_reward': combined_reward
            }
            results.append(result)
        
        return results


def load_dataset(data_path):
    """
    Load the Amazon dataset
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        DataFrame with the dataset
    """
    try:
        if data_path.endswith('.zip'):
            df = pd.read_csv(data_path, compression='zip')
        else:
            df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

