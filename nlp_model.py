"""
NLP Model for Product Description Generation using FLAN-T5.

The helpers in this module keep the Streamlit app flexible: they can build a
prompt from a hand-written product input or from different dataset schemas.
"""
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import warnings
import os
import re
from io import BytesIO, StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ProductDescriptionGenerator:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        """Initialize the instruction-tuned FLAN-T5 model and tokenizer."""
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize ROUGE scorer
        self.rouge = Rouge()
        
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
        
        print("Model loaded successfully!")
    
    def generate_text(self, prompt, max_new_tokens=150, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generate text using the instruction-tuned model.
        
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
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            
            with torch.no_grad():
                generation_args = {
                    **encoded,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,
                    "num_beams": 4,
                    "do_sample": False,
                    "repetition_penalty": 1.25,
                    "no_repeat_ngram_size": 3,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "early_stopping": True,
                    "length_penalty": 1.0,
                }
                outputs = self.model.generate(**generation_args)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = clean_generated_text(generated_text)

            if is_low_quality_generation(generated_text, prompt):
                generated_text = fallback_description_from_prompt(prompt)

            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""

    def generate_product_description(
        self,
        product_name,
        product_details,
        category=None,
        extra_details=None,
        max_new_tokens=150,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    ):
        """Generate a polished product description from flexible inputs."""
        prompt = build_product_prompt(
            product_name=product_name,
            product_details=product_details,
            category=category,
            extra_details=extra_details
        )
        return self.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
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
        if hasattr(data_path, "name"):
            data_path.seek(0)
            file_name = data_path.name.lower()
            compression = "zip" if file_name.endswith(".zip") else "infer"
            df = pd.read_csv(data_path, compression=compression)
        elif isinstance(data_path, (BytesIO, StringIO)):
            df = pd.read_csv(data_path)
        elif str(data_path).endswith('.zip'):
            df = pd.read_csv(data_path, compression='zip')
        else:
            df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def clean_text(value, fallback=""):
    """Return a safe, readable string for prompt building and metrics."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).replace("|", ". ").replace("\n", " ").strip()
    return " ".join(text.split())


def clean_generated_text(text):
    """Remove prompt labels and obvious broken repetitions from model output."""
    text = clean_text(text)
    if not text:
        return ""
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)

    blocked_prefixes = (
        "product name:",
        "category:",
        "product details:",
        "extra context:",
        "product description:",
    )
    sentences = []
    seen = set()
    for sentence in text.replace("\r", " ").split("."):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        lowered = sentence.lower()
        if lowered.startswith(blocked_prefixes):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        sentences.append(sentence)

    cleaned = ". ".join(sentences).strip()
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def is_low_quality_generation(text, prompt=None):
    """Detect empty, label-heavy, or repetitive generation."""
    cleaned = clean_text(text)
    if len(cleaned.split()) < 35:
        return True
    if count_sentences(cleaned) < 2:
        return True

    lowered = cleaned.lower()
    prompt_label_hits = sum(
        lowered.count(label)
        for label in (
            "product name:",
            "category:",
            "product details:",
            "extra context:",
            "product description:",
        )
    )
    if prompt_label_hits:
        return True

    words = lowered.split()
    if len(words) >= 24 and len(set(words)) / len(words) < 0.35:
        return True

    fourgrams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    if any(fourgrams.count(gram) > 1 for gram in set(fourgrams)):
        return True

    if prompt and not is_relevant_to_prompt(cleaned, prompt):
        return True

    return False


def is_relevant_to_prompt(text, prompt):
    """Check that generated copy stayed connected to the product input."""
    generated_words = set(simple_words(text))
    prompt_words = prompt_keywords(prompt)
    if not prompt_words:
        return True

    name_words = {
        word for word in simple_words(prompt_field(prompt, "Product name"))
        if word not in {"the", "and", "for", "with", "product"}
    }
    if name_words and not generated_words.intersection(name_words):
        return False

    matches = generated_words.intersection(prompt_words)
    required_matches = min(5, max(3, len(prompt_words) // 3))
    if len(matches) < required_matches:
        return False

    return not contains_wrong_product_type(text, prompt)


def contains_wrong_product_type(text, prompt):
    """Catch common ecommerce product swaps such as mouse -> keyboard."""
    generated_words = set(simple_words(text))
    prompt_words = set(simple_words(prompt))
    product_types = {
        "mouse", "keyboard", "headphone", "headphones", "speaker", "cable",
        "charger", "phone", "laptop", "watch", "camera", "monitor", "adapter",
        "earbuds", "tablet", "printer", "router",
    }
    prompt_product_words = prompt_words.intersection(product_types)
    generated_product_words = generated_words.intersection(product_types)
    wrong_words = generated_product_words - prompt_product_words
    return bool(prompt_product_words and wrong_words)


def simple_words(text):
    """Extract simple lowercase words without pulling in another dependency."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in clean_text(text))
    return [word for word in cleaned.split() if len(word) > 2]


def prompt_keywords(prompt):
    """Pull meaningful product words from the prompt for relevance checks."""
    stop_words = {
        "the", "and", "for", "with", "this", "that", "from", "into", "product",
        "description", "details", "category", "extra", "context", "write",
        "clear", "helpful", "persuasive", "ecommerce", "price", "between",
        "use", "uses", "using", "any", "one", "two", "all", "has", "have",
    }
    fields = [
        prompt_field(prompt, "Product name"),
        prompt_field(prompt, "Category"),
        prompt_field(prompt, "Product details"),
        prompt_field(prompt, "Extra context"),
    ]
    words = []
    for field in fields:
        words.extend(word for word in simple_words(field) if word not in stop_words)
    return set(words)


def prompt_field(prompt, label):
    """Extract a label value from a prompt built by build_product_prompt."""
    prefix = f"{label}:"
    for line in prompt.splitlines():
        if line.lower().startswith(prefix.lower()):
            return clean_text(line.split(":", 1)[1])
    return ""


def fallback_description_from_prompt(prompt):
    """Create a reliable description when the model repeats, drifts, or is too terse."""
    name = title_first_word(prompt_field(prompt, "Product name") or "This product")
    category = prompt_field(prompt, "Category")
    details = prompt_field(prompt, "Product details")
    extra = describe_extra_context(prompt_field(prompt, "Extra context"))
    audience = infer_audience(extra)

    category_phrase = f" {category}" if category else ""
    audience_phrase = f" for {audience}" if audience else ""

    intro = f"{name} is a smart{category_phrase} choice{audience_phrase}, built for smooth everyday use and reliable performance."
    detail_sentence = f"It includes {format_feature_list(details)}, giving you comfort, control, and convenience in one compact device." if details else ""
    extra_sentence = f"With {extra}, it fits buyers who want useful features without overspending." if extra else ""
    closing = "Its easy setup and practical design make it a dependable pick for work, browsing, study, or casual play."

    return " ".join(sentence for sentence in [intro, detail_sentence, extra_sentence, closing] if sentence)


def title_first_word(text):
    """Make short user-entered product names read better in generated copy."""
    text = clean_text(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def describe_extra_context(text):
    """Turn terse notes into a smoother phrase for the fallback description."""
    text = clean_text(text)
    if not text:
        return ""
    lowered = text.lower()
    lowered = lowered.replace("the price is between", "a price between")
    lowered = lowered.replace("price is between", "a price between")
    lowered = lowered.replace(" and is for gaming", " and gaming use")
    lowered = lowered.replace(" is for gaming", " for gaming use")
    return lowered


def infer_audience(text):
    """Infer a short audience phrase from optional user notes."""
    lowered = clean_text(text).lower()
    if "gaming" in lowered or "gamer" in lowered:
        return "gaming"
    if "student" in lowered:
        return "students"
    if "office" in lowered or "work" in lowered:
        return "office work"
    return ""


def sentence_case(text):
    """Make feature text fit naturally inside a sentence."""
    text = clean_text(text)
    if not text:
        return text
    return text[0].lower() + text[1:]


def format_feature_list(text):
    """Normalize dense product facts into readable feature copy."""
    text = clean_text(text)
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = text.replace("(", " (").replace(")", ") ")
    text = " ".join(text.split())
    return sentence_case(text)


def count_sentences(text):
    """Count sentence-like endings for quality checks."""
    return len(re.findall(r"[.!?](?:\s|$)", clean_text(text)))


def first_existing_column(columns, candidates):
    """Find the first candidate column name present in a dataset."""
    normalized = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower().strip())
        if match:
            return match
    return None


def suggest_columns(df):
    """Suggest useful product columns for common ecommerce datasets."""
    columns = df.columns.tolist()
    return {
        "name": first_existing_column(
            columns,
            ["product_name", "name", "title", "product_title", "item_name"]
        ),
        "description": first_existing_column(
            columns,
            ["about_product", "description", "product_description", "details", "features"]
        ),
        "category": first_existing_column(
            columns,
            ["category", "product_category", "main_category", "type"]
        ),
        "reference": first_existing_column(
            columns,
            ["review_content", "reviews", "review", "reference_text", "summary"]
        )
    }


def build_product_prompt(product_name, product_details, category=None, extra_details=None):
    """Build a consistent prompt for any product-like input."""
    name = clean_text(product_name, "Unnamed product")
    details = clean_text(product_details)
    category_text = clean_text(category)
    extra_text = clean_text(extra_details)

    prompt_parts = [
        "Write one polished ecommerce product description in 3 short sentences.",
        "Use the facts below. Do not repeat field labels. Do not invent technical specs.",
        "Return only the final product description.",
        f"Product name: {name}",
    ]
    if category_text:
        prompt_parts.append(f"Category: {category_text}")
    if details:
        prompt_parts.append(f"Product details: {details}")
    if extra_text:
        prompt_parts.append(f"Extra context: {extra_text}")
    prompt_parts.append("Answer:")
    return "\n".join(prompt_parts)


def row_to_prompt(row, name_col, description_col, category_col=None, extra_cols=None):
    """Build a generation prompt from a dataframe row and selected columns."""
    extra_cols = extra_cols or []
    extra_details = []
    for col in extra_cols:
        value = clean_text(row.get(col))
        if value:
            extra_details.append(f"{col}: {value}")

    return build_product_prompt(
        product_name=row.get(name_col),
        product_details=row.get(description_col),
        category=row.get(category_col) if category_col else None,
        extra_details="; ".join(extra_details)
    )

