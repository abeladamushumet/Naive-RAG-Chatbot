import pandas as pd
import re
import os

def load_data(filepath):
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    """Basic text cleaning: lowercase, remove special chars, boilerplate."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove boilerplate phrases common in complaints 
    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"this is a complaint regarding",
        r"please investigate this issue",
        r"thank you for your attention",
        r"i would like to report",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
    # Remove non-alphanumeric chars except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def filter_valid_products(df, valid_products):
    """Filter DataFrame to keep only valid products and non-empty narratives."""
    filtered_df = df[df['Product'].isin(valid_products)].copy()
    filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notnull()]
    filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].str.strip() != ""]
    return filtered_df

def preprocess_dataset(input_csv, output_csv, valid_products):
    """Complete preprocessing pipeline."""
    print(f"Loading data from {input_csv}...")
    df = load_data(input_csv)
    print(f"Original dataset size: {len(df):,}")

    print("Filtering valid products and non-empty narratives...")
    df_filtered = filter_valid_products(df, valid_products)
    print(f"Filtered dataset size: {len(df_filtered):,}")

    print("Cleaning complaint narratives...")
    df_filtered['Cleaned Narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)

    # Remove rows where cleaning results in empty narrative
    df_filtered = df_filtered[df_filtered['Cleaned Narrative'].str.strip() != ""]
    print(f"Dataset size after cleaning empty narratives: {len(df_filtered):,}")

    print(f"Saving cleaned data to {output_csv}...")
    df_filtered.to_csv(output_csv, index=False)
    print("Preprocessing complete.")

def resolve_path(p):
    """Resolve to absolute path if relative."""
    return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)

if __name__ == "__main__":
    import sys

    print("Current working directory:", os.getcwd())

    raw_data_path = sys.argv[1] if len(sys.argv) > 1 else "Data/raw/complaints.csv"
    raw_data_path = resolve_path(raw_data_path)

    output_path = sys.argv[2] if len(sys.argv) > 2 else "Data/processed/filtered_complaints.csv"
    output_path = resolve_path(output_path)

    print(f"Using input data path: {raw_data_path}")
    print(f"Using output data path: {output_path}")

    valid_products = [
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later",
        "Savings account",
        "Money transfer, virtual currency"
    ]

    preprocess_dataset(raw_data_path, output_path, valid_products)
