import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for all plots
sns.set(style="whitegrid")

def load_data(file_path):
    """
    Load the complaint dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


def plot_product_distribution(df, product_col="Product"):
    """
    Plot the distribution of complaints by product.

    Parameters:
        df (pd.DataFrame): Complaint data.
        product_col (str): Column name for the product field.
    """
    product_counts = df[product_col].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=product_counts.index, y=product_counts.values, palette="Blues_d")
    plt.title("Complaint Count by Product Category")
    plt.ylabel("Number of Complaints")
    plt.xlabel("Product")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_narrative_length_distribution(df, narrative_col="Consumer complaint narrative"):
    """
    Plot a histogram showing distribution of narrative lengths (in words).

    Parameters:
        df (pd.DataFrame): Complaint data.
        narrative_col (str): Column name for the complaint text.
    """
    df = df.copy()
    df["narrative_length"] = df[narrative_col].dropna().astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df["narrative_length"], bins=50, kde=True, color="orange")
    plt.title("Distribution of Narrative Lengths (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def get_narrative_stats(df, narrative_col="Consumer complaint narrative"):
    """
    Get statistics about complaint narratives.

    Parameters:
        df (pd.DataFrame): Complaint data.
        narrative_col (str): Column name for the complaint text.

    Returns:
        dict: Summary statistics.
    """
    df = df.copy()
    df["narrative_length"] = df[narrative_col].dropna().astype(str).apply(lambda x: len(x.split()))

    stats = {
        "Total complaints": len(df),
        "With narratives": df[narrative_col].notnull().sum(),
        "Without narratives": df[narrative_col].isnull().sum(),
        "Average word count": df["narrative_length"].mean(),
        "Max word count": df["narrative_length"].max(),
        "Min word count": df["narrative_length"].min()
    }

    return stats


def filter_valid_products(df, valid_products, product_col="Product", narrative_col="Consumer complaint narrative"):
    """
    Filter dataset for selected products and non-empty complaint narratives.

    Parameters:
        df (pd.DataFrame): Complaint data.
        valid_products (list): List of allowed product names.
        product_col (str): Product column name.
        narrative_col (str): Complaint text column name.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df[product_col].isin(valid_products)) &
        (df[narrative_col].notnull()) &
        (df[narrative_col].str.strip() != "")
    ].copy()
    return filtered_df.reset_index(drop=True)