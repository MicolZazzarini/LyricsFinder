import pandas as pd
import random
from nltk.corpus import wordnet as wn
from transformers import MarianMTModel, MarianTokenizer
import torch
from datetime import datetime
import os


def clean_data(data):
    """
    Cleans the dataset by removing rows with missing or invalid values, 
    and standardizes the lyrics column.

    Args:
        data (DataFrame): Input dataset with 'genre' and 'lyrics' columns.

    Returns:
        DataFrame: Cleaned dataset.
    """

    data = data.dropna(subset=['genre', 'lyrics'])
    data = data[data['genre'].str.lower() != 'not available']
    data['lyrics'] = data['lyrics'].str.strip()
    data['lyrics'] = data['lyrics'].str.replace(r"[-\?.,\/#!$%\^&\*;:{}=\_~()]", ' ', regex=True)
    data['lyrics'] = data['lyrics'].str.replace(r"\[(.*?)\]", ' ', regex=True)
    data['lyrics'] = data['lyrics'].str.replace(r"' | '", ' ', regex=True)
    data['lyrics'] = data['lyrics'].str.replace(r"x[0-9]+", ' ', regex=True)
    data = data[data['lyrics'].str.strip().str.lower() != 'instrumental']
    data = data[~data['lyrics'].str.contains(r'[^\x00-\x7F]+')]
    data = data[data['lyrics'].str.strip() != '']
    data = data[data['genre'].str.lower() != 'not available']

    return data


def filter_genres(data):
    """
    Filters the dataset to include only specific genres.

    Args:
        data (DataFrame): Input dataset with 'genre' column.

    Returns:
        DataFrame: Filtered dataset.
    """
    data = data.loc[(data['genre'] == 'Pop') |
                    (data['genre'] == 'Rock') |
                    (data['genre'] == 'Hip-Hop')|
                    (data['genre'] == 'Metal')
                ]
    return data


def undersample_data(data, max_samples_per_genre):
    """
    Balances the dataset by undersampling genres to a specified limit.

    Args:
        data (DataFrame): Input dataset with 'genre' column.
        max_samples_per_genre (int): Maximum samples per genre.

    Returns:
        DataFrame: Balanced dataset.
    """
    balanced_samples = []
    genres = data['genre'].unique()

    for genre in genres:
        genre_subset = data[data['genre'] == genre]
        if len(genre_subset) >= max_samples_per_genre:
            balanced_samples.append(genre_subset.sample(n=max_samples_per_genre, random_state=42))
        else:
            print(f"Not enough samples for genre '{genre}', only {len(genre_subset)} available.")
            balanced_samples.append(genre_subset.sample(n=len(genre_subset), random_state=42))

    balanced_data = pd.concat(balanced_samples, ignore_index=True)
    return balanced_data


def synonym_augmentation(text):
    """
    Augments text by replacing words with their synonyms using WordNet.

    Args:
        text (str): Input text to be augmented.

    Returns:
        str: Augmented text.
    """
    words = text.split()
    augmented_text = []
    for word in words:
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        if len(synonyms) > 1:
            synonyms.discard(word)
            augmented_text.append(random.choice(list(synonyms)))
        else:
            augmented_text.append(word)

    return " ".join(augmented_text)


def back_translate_batch(texts, src_lang='en', tgt_lang='fr', batch_size=16):
    """
    Performs back-translation on a batch of texts using a MarianMT model.

    Args:
        texts (list of str): List of texts to translate.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        batch_size (int): Batch size for translation.

    Returns:
        list of str: Back-translated texts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model_back = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}").to(device)
    tokenizer_back = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}")

    back_translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=256)
        translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        back_inputs = tokenizer_back(translated_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        back_outputs = model_back.generate(**back_inputs, max_length=256)
        back_translated_texts.extend([tokenizer_back.decode(output, skip_special_tokens=True) for output in back_outputs])

    return back_translated_texts


def augment_data(data, target_per_class):
    """
    Augments data to meet the target sample count per genre.

    Args:
        data (DataFrame): Input dataset with 'lyrics' and 'genre' columns.
        target_per_class (dict): Target sample count per genre.

    Returns:
        DataFrame: Dataset with augmented data.
    """
    augmented_data = []
    genres = target_per_class.keys()

    for genre in genres:
        genre_data = data[data['genre'] == genre]
        current_count = len(genre_data)
        samples_needed = max(0, target_per_class[genre] - current_count)

        if samples_needed > 0:
            original_texts = genre_data['lyrics'].sample(n=samples_needed, replace=True).tolist()

            back_translated_texts = back_translate_batch(original_texts, batch_size=16)

            augmented_texts = [synonym_augmentation(text) for text in back_translated_texts]

            augmented_data.extend([[text, genre] for text in augmented_texts])

    augmented_df = pd.DataFrame(augmented_data, columns=['lyrics', 'genre'])
    return pd.concat([data, augmented_df], ignore_index=True)


def save_cleaned_data(data, output_dir = "results/cleaning"):
    """
    Saves the cleaned data to a timestamped directory.

    Args:
        data (DataFrame): Cleaned dataset to save.
        output_dir (str): Directory to save the cleaned data.

    Returns:
        file_path (str): path of the cleaned dataset
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"cleaned_data_{timestamp}.csv")
    data.to_csv(file_path, index=False)
    print(f"Cleaned data saved in: {file_path}")
    return file_path


def cleaning(input_path, max_samples_per_genre, target_per_class, output_dir="results/cleaning"):
    """
    Function to perform the entire cleaning and augmentation process

    Args:
        input_path (str): Path to the input CSV file.
        max_samples_per_genre (int): Maximum samples per genre for balancing.
        target_per_class (dict): Target sample count per genre for augmentation.
        output_dir (str): Directory to save the processed data.

    Returns:
        file_path (str): path of the cleaned dataset
    """
    # Load the dataset
    data = pd.read_csv(input_path)
    print("Dataset loaded.")

    # Clean the data
    cleaned_data = clean_data(data)
    print("Data cleaned.")

    # Filter for specific genres
    filtered_data = filter_genres(cleaned_data)
    print("Genres filtered.")

    # Balance the dataset using undersampling
    balanced_data = undersample_data(filtered_data, max_samples_per_genre)
    print("Data balanced with undersampling.")

    # Augment the data to meet target samples per genre
    augmented_data = augment_data(balanced_data, target_per_class)
    print("Data augmented.")

    # Save the cleaned and processed data
    return save_cleaned_data(augmented_data, output_dir=output_dir)


if __name__ == "__main__":
    input_path = r'..\dataset\lyrics.csv'
    max_samples_per_genre = 300
    target_per_class = {
        'Pop': 301,
        'Rock': 301,
        'Hip-Hop': 301,
        'Metal': 301
    }

    # Call the cleaning function
    cleaning(input_path, max_samples_per_genre, target_per_class)

