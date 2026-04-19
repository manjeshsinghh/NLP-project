"""
Streamlit app for flexible product description generation.
"""
import os

import pandas as pd
import streamlit as st

from nlp_model import (
    ProductDescriptionGenerator,
    clean_text,
    load_dataset,
    row_to_prompt,
    suggest_columns,
)


st.set_page_config(
    page_title="Product Description Generator",
    page_icon=":shopping_bags:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #176b87;
        text-align: center;
        margin-bottom: 1rem;
    }
    .generated-text {
        background-color: #eef7f2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d8f6f;
        color: #1f2933;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the GPT-2 model once per Streamlit session."""
    return ProductDescriptionGenerator()


@st.cache_data(show_spinner=False)
def load_dataset_cached(path):
    """Cache local dataset loading by path."""
    return load_dataset(path)


def option_index(options, value, default=0):
    if value in options:
        return options.index(value)
    return default


def show_metrics(generator, generated_text, reference_text):
    reference_text = clean_text(reference_text)
    if not reference_text:
        st.info("No reference text was provided, so automatic evaluation was skipped.")
        return

    reward_scores = generator.reward_function(generated_text, reference_text)
    metric_cols = st.columns(5)
    metric_cols[0].metric("BLEU", f"{reward_scores['bleu']:.4f}")
    metric_cols[1].metric("ROUGE-1", f"{reward_scores['rouge_1']:.4f}")
    metric_cols[2].metric("ROUGE-2", f"{reward_scores['rouge_2']:.4f}")
    metric_cols[3].metric("ROUGE-L", f"{reward_scores['rouge_l']:.4f}")
    metric_cols[4].metric("Combined", f"{reward_scores['combined']:.4f}")


def load_dataset_from_ui(dataset_path, uploaded_file, load_data):
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is None or df.empty:
            st.error("The uploaded dataset could not be loaded.")
            return None
        st.success(f"Uploaded dataset loaded: {len(df)} rows")
        return df

    if load_data:
        if not os.path.exists(dataset_path):
            st.error(f"Dataset file not found: {dataset_path}")
            return None
        df = load_dataset_cached(dataset_path)
        if df is None or df.empty:
            st.error("The dataset could not be loaded or is empty.")
            return None
        st.success(f"Dataset loaded: {len(df)} rows")
        return df

    return st.session_state.get("dataset")


def main():
    st.markdown('<h1 class="main-header">Product Description Generator</h1>', unsafe_allow_html=True)
    st.caption("Generate descriptions from one product, any dataset row, or a small batch of products.")

    with st.sidebar:
        st.header("Configuration")
        max_tokens = st.slider("Max new tokens", 40, 300, 140, 10)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.75, 0.05)
        top_k = st.slider("Top-K", 1, 100, 50, 5)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)

        st.divider()
        st.subheader("Dataset")
        dataset_path = st.text_input("Local dataset path", value="amazon.csv.zip")
        uploaded_file = st.file_uploader("Or upload CSV/ZIP", type=["csv", "zip"])
        load_data = st.button("Load dataset", type="primary", use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Single product", "Dataset products", "About"])

    with tab1:
        st.header("Generate for any product")
        with st.spinner("Loading model..."):
            generator = load_model()

        col1, col2 = st.columns(2)
        with col1:
            product_name = st.text_input(
                "Product name",
                placeholder="Wireless Bluetooth Headphones",
            )
            category = st.text_input(
                "Category",
                placeholder="Electronics, home, fashion, beauty...",
            )
        with col2:
            product_description = st.text_area(
                "Product details",
                placeholder="Noise cancellation, 30-hour battery, lightweight design...",
                height=120,
            )

        extra_details = st.text_area(
            "Extra details",
            placeholder="Target audience, tone, warranty, price, keywords, or anything else to include",
            height=90,
        )
        reference_text = st.text_area(
            "Reference text for optional evaluation",
            placeholder="Paste a review or ideal description if you want BLEU/ROUGE scores",
            height=90,
        )

        if st.button("Generate description", type="primary", use_container_width=True):
            if not clean_text(product_name) or not clean_text(product_description):
                st.error("Please enter at least a product name and product details.")
            else:
                with st.spinner("Generating description..."):
                    generated_text = generator.generate_product_description(
                        product_name=product_name,
                        product_details=product_description,
                        category=category,
                        extra_details=extra_details,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                st.subheader("Generated description")
                st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
                show_metrics(generator, generated_text, reference_text)

    with tab2:
        st.header("Use any product from a dataset")

        df = load_dataset_from_ui(dataset_path, uploaded_file, load_data)
        if df is not None:
            st.session_state.dataset = df

        if st.session_state.get("dataset") is None:
            st.info("Load the included Amazon dataset or upload your own CSV/ZIP file to begin.")
        else:
            df = st.session_state.dataset.copy()
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            with st.expander("Preview dataset", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

            suggestions = suggest_columns(df)
            columns = df.columns.tolist()
            no_column = "(none)"
            selectable_columns = [no_column] + columns

            st.subheader("Map your columns")
            col1, col2, col3 = st.columns(3)
            with col1:
                name_col = st.selectbox(
                    "Product name column",
                    columns,
                    index=option_index(columns, suggestions["name"]),
                )
            with col2:
                description_col = st.selectbox(
                    "Product details column",
                    columns,
                    index=option_index(columns, suggestions["description"]),
                )
            with col3:
                category_choice = st.selectbox(
                    "Category column",
                    selectable_columns,
                    index=option_index(selectable_columns, suggestions["category"]),
                )
                category_col = None if category_choice == no_column else category_choice

            reference_choice = st.selectbox(
                "Reference/evaluation column",
                selectable_columns,
                index=option_index(selectable_columns, suggestions["reference"]),
            )
            reference_col = None if reference_choice == no_column else reference_choice

            extra_cols = st.multiselect(
                "Extra columns to include in the prompt",
                [col for col in columns if col not in {name_col, description_col, category_col, reference_col}],
                default=[],
            )

            search_query = st.text_input("Search products", placeholder="Type part of a product name or category")
            filtered_df = df
            if clean_text(search_query):
                query = search_query.lower()
                search_cols = [name_col, description_col]
                if category_col:
                    search_cols.append(category_col)
                mask = pd.Series(False, index=df.index)
                for col in search_cols:
                    mask = mask | df[col].astype(str).str.lower().str.contains(query, na=False, regex=False)
                filtered_df = df[mask]

            if filtered_df.empty:
                st.warning("No matching products found.")
            else:
                filtered_indexes = filtered_df.index.tolist()
                selected_position = st.selectbox(
                    "Choose any product",
                    range(len(filtered_indexes)),
                    format_func=lambda position: (
                        f"Row {filtered_indexes[position]}: "
                        f"{clean_text(filtered_df.iloc[position].get(name_col), 'Unnamed product')[:90]}"
                    ),
                )
                selected_index = filtered_indexes[selected_position]
                selected_row = df.loc[selected_index]

                st.markdown("#### Selected product")
                st.write(clean_text(selected_row.get(name_col), "Unnamed product"))
                st.caption(clean_text(selected_row.get(description_col))[:500])

                with st.spinner("Loading model..."):
                    generator = load_model()

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate selected product", type="primary", use_container_width=True):
                        prompt = row_to_prompt(
                            selected_row,
                            name_col=name_col,
                            description_col=description_col,
                            category_col=category_col,
                            extra_cols=extra_cols,
                        )
                        with st.spinner("Generating description..."):
                            generated_text = generator.generate_text(
                                prompt,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                        st.subheader("Generated description")
                        st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
                        if reference_col:
                            show_metrics(generator, generated_text, selected_row.get(reference_col))

                with col2:
                    batch_size = st.number_input(
                        "Batch size",
                        min_value=1,
                        max_value=min(25, len(filtered_df)),
                        value=min(5, len(filtered_df)),
                        step=1,
                        help="Small batches are recommended because GPT-2 generation is slow on CPU.",
                    )
                    if st.button("Generate batch", use_container_width=True):
                        batch_rows = filtered_df.head(int(batch_size))
                        results = []
                        progress = st.progress(0)
                        for position, (idx, row) in enumerate(batch_rows.iterrows(), start=1):
                            prompt = row_to_prompt(
                                row,
                                name_col=name_col,
                                description_col=description_col,
                                category_col=category_col,
                                extra_cols=extra_cols,
                            )
                            generated_text = generator.generate_text(
                                prompt,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                            result = {
                                "row_index": idx,
                                "product_name": clean_text(row.get(name_col), "Unnamed product"),
                                "generated_description": generated_text,
                            }
                            if reference_col:
                                scores = generator.reward_function(generated_text, clean_text(row.get(reference_col)))
                                result["combined_score"] = scores["combined"]
                            results.append(result)
                            progress.progress(position / len(batch_rows))

                        results_df = pd.DataFrame(results)
                        st.subheader("Batch results")
                        st.dataframe(results_df, use_container_width=True)
                        st.download_button(
                            "Download generated descriptions",
                            data=results_df.to_csv(index=False).encode("utf-8"),
                            file_name="generated_product_descriptions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

    with tab3:
        st.header("About this project")
        st.markdown(
            """
            This app uses GPT-2 to generate ecommerce product descriptions from flexible product input.

            What improved:

            - The dataset tab can use any row, not only the first few products.
            - You can upload a CSV/ZIP dataset or load the included Amazon dataset.
            - Column mapping lets the app work with datasets that use different names.
            - Batch generation can create descriptions for several products at once.
            - The prompt now includes product name, details, category, and optional extra fields.
            """
        )


if __name__ == "__main__":
    main()
