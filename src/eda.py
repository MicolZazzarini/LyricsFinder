import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_results_directory(base_dir="results/eda"):
    """
    Creates a directory for saving results, with a timestamp to ensure uniqueness.

    Args:
        base_dir (str): Base directory where the results folder will be created.

    Returns:
        str: Path to the created directory.
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Generate a unique directory name using the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, f"data_visualization_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results directory created at: {results_dir}")
    return results_dir


def total_samples(data, results_dir):
    """
    Calculates and saves the total number of samples in the dataset.

    Args:
        data (pd.DataFrame): The dataset.
        results_dir (str): Directory where the results will be saved.

    Returns:
        int: Total number of samples.
    """
    total_samples = data.shape[0]
    result = f"Total number of samples in the dataset: {total_samples}"
    print(result)

    # Save the result to a file
    with open(os.path.join(results_dir, 'total_samples.txt'), 'w') as f:
        f.write(result)

    return total_samples


def unique_genres(data, results_dir):
    """
    Identifies and saves the unique genres in the dataset.

    Args:
        data (pd.DataFrame): The dataset.
        results_dir (str): Directory where the results will be saved.

    Returns:
        np.ndarray: Array of unique genres.
    """
    unique_genres = data['genre'].unique()
    result = f"Unique genres in the dataset:\n{unique_genres}"
    print(result)

    # Save the result to a file
    with open(os.path.join(results_dir, 'unique_genres.txt'), 'w') as f:
        f.write(result)

    return unique_genres


def genre_distribution(data, results_dir):
    """
    Calculates and saves the genre distribution and the corresponding plot.

    Args:
        data (pd.DataFrame): The dataset.
        results_dir (str): Directory where the results will be saved.

    Returns:
        pd.Series: Genre distribution.
    """
    genre_distribution = data['genre'].value_counts()
    result = "\nGenre distribution (number of samples per genre):\n" + genre_distribution.to_string()
    print(result)

    # Save the distribution to a file
    with open(os.path.join(results_dir, 'genre_distribution.txt'), 'w') as f:
        f.write(result)

    # Create and save the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_distribution.index, y=genre_distribution.values, palette="viridis")
    plt.title("Genre Distribution in the Dataset")
    plt.xlabel("Genres")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.savefig(os.path.join(results_dir, 'genre_distribution.png'))
    plt.close()

    return genre_distribution


def perform_eda(data, base_dir="results/eda"):
    """
    Executes all EDA operations and saves the results in a created directory.

    Args:
        data (pd.DataFrame): The dataset.
        base_dir (str): Base directory where the results folder will be created.
    """
    print("Starting EDA...")

    # Create the results directory
    results_dir = create_results_directory(base_dir)

    # Perform EDA tasks
    total_samples(data, results_dir)
    unique_genres(data, results_dir)
    genre_distribution(data, results_dir)

    print("EDA completed. Results saved.")


if __name__ == "__main__":
    # Dataset path
    data_path = r'results\cleaning\cleaned_data_20241224_112608.csv'

    # Load the dataset
    data = pd.read_csv(data_path)

    # Perform EDA
    perform_eda(data)
