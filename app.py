"""
Streamlit App for Product Description Generation
Interactive web interface for the NLP project
"""
import streamlit as st
import pandas as pd
import os
from nlp_model import ProductDescriptionGenerator, load_dataset

# Page configuration
st.set_page_config(
    page_title="Product Description Generator",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .generated-text {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the GPT-2 model (cached for performance)"""
    return ProductDescriptionGenerator()

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Product Description Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model parameters
        st.subheader("Generation Parameters")
        max_tokens = st.slider("Max New Tokens", 50, 300, 150, 10)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_k = st.slider("Top-K", 1, 100, 50, 5)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
        
        st.markdown("---")
        
        # Iteration settings
        st.subheader("Feedback Loop Settings")
        num_iterations = st.slider("Number of Iterations", 1, 10, 5, 1)
        
        st.markdown("---")
        
        # Dataset selection
        st.subheader("Dataset")
        dataset_path = st.text_input("Dataset Path", value="amazon.csv.zip")
        load_data = st.button("Load Dataset", type="primary")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            st.warning(f"⚠️ Dataset file not found: {dataset_path}")
            st.info("💡 Make sure the dataset file is in the same directory as app.py")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Single Product", "📊 Batch Processing", "📈 About"])
    
    # Tab 1: Single Product Generation
    with tab1:
        st.header("Generate Description for Single Product")
        
        # Load model
        with st.spinner("Loading GPT-2 model..."):
            generator = load_model()
        st.success("Model loaded successfully!")
        
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input(
                "Product Name",
                value="Wayona Nylon Braided USB to Lightning Fast Charging Cable",
                help="Enter the name of the product"
            )
        
        with col2:
            product_description = st.text_area(
                "Product Description",
                value="Fast charging and data sync cable compatible with iPhone devices",
                help="Enter a brief description of the product",
                height=100
            )
        
        reference_text = st.text_area(
            "Reference Text (for evaluation)",
            value="Great cable, fast charging, durable build quality",
            help="Enter reference text for evaluation metrics (BLEU, ROUGE)",
            height=100
        )
        
        # Generate button
        if st.button("Generate Description", type="primary", use_container_width=True):
            if not product_name or not product_description:
                st.error("Please fill in both Product Name and Product Description")
            else:
                # Create prompt
                prompt = f"Product Name: {product_name}\nDescription: {product_description}\nGenerate a compelling product description:"
                
                # Generate text
                with st.spinner("Generating description..."):
                    generated_text = generator.generate_text(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                
                # Display generated text
                st.markdown("### Generated Description")
                st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
                
                # Calculate metrics
                if reference_text:
                    with st.spinner("Calculating evaluation metrics..."):
                        reward_scores = generator.reward_function(generated_text, reference_text)
                    
                    # Display metrics
                    st.markdown("### Evaluation Metrics")
                    metric_cols = st.columns(5)
                    
                    with metric_cols[0]:
                        st.metric("BLEU Score", f"{reward_scores['bleu']:.4f}")
                    with metric_cols[1]:
                        st.metric("ROUGE-1", f"{reward_scores['rouge_1']:.4f}")
                    with metric_cols[2]:
                        st.metric("ROUGE-2", f"{reward_scores['rouge_2']:.4f}")
                    with metric_cols[3]:
                        st.metric("ROUGE-L", f"{reward_scores['rouge_l']:.4f}")
                    with metric_cols[4]:
                        st.metric("Combined", f"{reward_scores['combined']:.4f}")
    
    # Tab 2: Batch Processing with Iterative Feedback
    with tab2:
        st.header("Batch Processing with Iterative Feedback")
        
        # Load dataset
        if 'dataset' not in st.session_state:
            st.session_state.dataset = None
        
        if load_data or st.session_state.dataset is not None:
            if st.session_state.dataset is None:
                if os.path.exists(dataset_path):
                    try:
                        with st.spinner("Loading dataset..."):
                            df = load_dataset(dataset_path)
                            if df is not None and not df.empty:
                                st.session_state.dataset = df
                                st.success(f"✅ Dataset loaded successfully! ({len(df)} products)")
                                st.info(f"📊 Columns: {', '.join(df.columns.tolist()[:5])}...")
                            else:
                                st.error("❌ Failed to load dataset or dataset is empty")
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {str(e)}")
                else:
                    st.error(f"❌ Dataset file not found: {dataset_path}")
                    st.info("💡 Please check the file path in the sidebar")
            
            if st.session_state.dataset is not None:
                df = st.session_state.dataset
                
                # Check for required columns
                required_columns = ['product_name', 'about_product', 'review_content']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info(f"📊 Available columns: {', '.join(df.columns.tolist())}")
                else:
                    # Product selector
                    product_options = df['product_name'].tolist() if 'product_name' in df.columns else []
                    if product_options:
                        # Limit to first 10 products for performance
                        num_products = min(10, len(df))
                        selected_product_idx = st.selectbox(
                            "Select Product",
                            range(num_products),
                            format_func=lambda x: f"Product {x+1}: {product_options[x][:50] if len(product_options[x]) > 50 else product_options[x]}..."
                        )
                        
                        if st.button("Process Product", type="primary"):
                            # Get product data
                            row = df.iloc[selected_product_idx]
                            product_name = row.get('product_name', 'N/A')
                            description = row.get('about_product', 'N/A')
                            reference_text = row.get('review_content', 'N/A')
                            
                            # Check for missing data
                            if pd.isna(product_name) or pd.isna(description):
                                st.error("❌ Selected product has missing required data (product_name or about_product)")
                            else:
                                # Create prompt
                                prompt = f"Product Name: {product_name}\nDescription: {description}\nGenerate a compelling product description:"
                                
                                # Load model
                                generator = load_model()
                                
                                # Manual feedback scores
                                st.markdown("### Iterative Feedback Loop")
                                st.markdown(f"**Product:** {product_name}")
                                st.markdown(f"**Description:** {description}")
                                if not pd.isna(reference_text):
                                    st.markdown(f"**Reference Text:** {reference_text}")
                                
                                manual_scores = []
                                
                                results_container = st.container()
                                
                                for i in range(num_iterations):
                                    with results_container:
                                        st.markdown(f"#### Iteration {i + 1}")
                                        
                                        # Generate text
                                        with st.spinner(f"Generating text for iteration {i + 1}..."):
                                            generated_text = generator.generate_text(
                                                prompt,
                                                max_new_tokens=max_tokens,
                                                temperature=temperature,
                                                top_k=top_k,
                                                top_p=top_p
                                            )
                                        
                                        # Display generated text
                                        st.markdown("**Generated Text:**")
                                        st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
                                        
                                        # Calculate metrics (only if reference text is available)
                                        if not pd.isna(reference_text) and reference_text.strip():
                                            reward_scores = generator.reward_function(generated_text, reference_text)
                                            
                                            # Display metrics
                                            metric_cols = st.columns(5)
                                            with metric_cols[0]:
                                                st.metric("BLEU", f"{reward_scores['bleu']:.4f}")
                                            with metric_cols[1]:
                                                st.metric("ROUGE-1", f"{reward_scores['rouge_1']:.4f}")
                                            with metric_cols[2]:
                                                st.metric("ROUGE-2", f"{reward_scores['rouge_2']:.4f}")
                                            with metric_cols[3]:
                                                st.metric("ROUGE-L", f"{reward_scores['rouge_l']:.4f}")
                                            with metric_cols[4]:
                                                st.metric("Combined", f"{reward_scores['combined']:.4f}")
                                        else:
                                            st.info("⚠️ No reference text provided. Skipping automatic evaluation.")
                                            reward_scores = {'combined': 0.0}
                                        
                                        # Manual feedback
                                        manual_score = st.slider(
                                            f"Rate iteration {i + 1} (1-10)",
                                            1, 10, 5,
                                            key=f"manual_score_{selected_product_idx}_{i}"
                                        )
                                        manual_scores.append(manual_score)
                                        
                                        # Combined reward
                                        combined_reward = (0.7 * manual_score / 10) + (0.3 * reward_scores['combined'])
                                        st.metric("Combined Reward (Manual + Automatic)", f"{combined_reward:.4f}")
                                        
                                        st.markdown("---")
                    else:
                        st.warning("⚠️ No products found in dataset")
    
    # Tab 3: About
    with tab3:
        st.header("About This Project")
        st.markdown("""
        ### Product Description Generator using GPT-2
        
        This application uses GPT-2 to generate compelling product descriptions based on product names and descriptions.
        
        **Features:**
        - 🤖 GPT-2 based text generation
        - 📊 Automatic evaluation using BLEU and ROUGE metrics
        - 🔄 Iterative feedback loop for improvement
        - 📈 Real-time metrics visualization
        
        **Metrics:**
        - **BLEU Score**: Measures n-gram precision between generated and reference text
        - **ROUGE-1**: Unigram overlap (precision, recall, F1)
        - **ROUGE-2**: Bigram overlap (precision, recall, F1)
        - **ROUGE-L**: Longest common subsequence based metrics
        
        **Usage:**
        1. Enter product name and description
        2. Optionally provide reference text for evaluation
        3. Adjust generation parameters in the sidebar
        4. Click "Generate Description" to create product descriptions
        5. View evaluation metrics and provide feedback
        
        **Technologies:**
        - Transformers (Hugging Face)
        - PyTorch
        - NLTK
        - ROUGE
        - Streamlit
        """)

if __name__ == "__main__":
    main()

