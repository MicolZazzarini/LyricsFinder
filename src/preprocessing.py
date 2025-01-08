import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import nltk
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime


# Initialize stopwords and lemmatizer
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Load the BERT tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class LyricsDataset(Dataset):
    """
    A custom Dataset class for handling lyrics and labels to work with DataLoader.

    Args:
        lyrics (list): A list of lyrics.
        labels (list): A list of corresponding genre labels.
        tokenizer (BertTokenizer): The tokenizer to preprocess text.
        max_len (int): The maximum length of tokenized input.
    """
    def __init__(self, lyrics, labels, tokenizer, max_len=128):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        """
        Returns tokenized text and its corresponding label.

        Args:
            item (int): Index of the sample.

        Returns:
            dict: A dictionary containing tokenized input and labels.
        """
        text = self.lyrics[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def remove_stopwords(text):
    """
    Removes stopwords from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without stopwords.
    """
    return ' '.join([word for word in text.split() if word not in stop])



def lemmatize_text(text):
    """
    Lemmatizes the given text by converting words to their base form.

    Args:
        text (str): The input text.

    Returns:
        str: The lemmatized text.
    """
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])



def preprocess_lyrics(data):
    """
    Preprocesses the lyrics by removing stopwords and lemmatizing the words.

    Args:
        data (DataFrame): The input DataFrame containing 'lyrics' column.

    Returns:
        DataFrame: The DataFrame with preprocessed lyrics.
    """
    data['lyrics'] = data['lyrics'].apply(remove_stopwords)
    data['lyrics'] = data['lyrics'].apply(lemmatize_text)
    return data



def encode_labels(data):
    """
    Encodes the genre labels into numeric values.

    Args:
        data (DataFrame): The input DataFrame containing 'genre' column.

    Returns:
        DataFrame, LabelEncoder: The DataFrame with encoded genre labels and the label encoder used.
    """
    label_encoder = LabelEncoder()
    data['encoded_genre'] = label_encoder.fit_transform(data['genre'])
    return data, label_encoder



def split_dataset(data, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (DataFrame): The input DataFrame containing lyrics and encoded genres.
        test_size (float): Proportion of the data to be used as test set.
        random_state (int): Seed for random number generator.

    Returns:
        tuple: Split datasets (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data['lyrics'],
        data['encoded_genre'],
        test_size=test_size,
        random_state=random_state,
        stratify=data['encoded_genre']
    )
    return X_train, X_test, y_train, y_test



def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir = "results/preprocessing"):
    """
    Saves the preprocessed dataset to CSV files.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Test features.
        y_train (DataFrame): Training labels.
        y_test (DataFrame): Test labels.
        output_dir (str): Directory to save the preprocessed data.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, f"preprocessed_data_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    X_train.to_csv(os.path.join(results_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(results_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
    print(f"Preprocessed data saved in: {results_dir}")



def save_dataloaders(train_loader, test_loader, output_dir="results/preprocessing"):
    """
    Saves the data loaders to disk.

    Args:
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.
        output_dir (str): Directory to save the data loaders.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataloaders_dir = os.path.join(output_dir, f"data_loaders")
    os.makedirs(dataloaders_dir, exist_ok=True)
    dataloaders_dir_1 = os.path.join(dataloaders_dir, f"dataloaders_{timestamp}")
    os.makedirs(dataloaders_dir_1, exist_ok=True)

    torch.save(train_loader, os.path.join(dataloaders_dir_1, 'train_loader.pth'))
    torch.save(test_loader, os.path.join(dataloaders_dir_1, 'test_loader.pth'))

    print("Data loaders saved.")



def preprocessing(input_path, batch_size=16, test_size=0.2, random_state=42, output_dir = "results/preprocessing"):
    """
    Preprocesses the dataset, splits it, and prepares DataLoader for training and testing.

    Args:
        input_path (str): Path to the input CSV file.
        batch_size (int): The batch size for DataLoader.
        test_size (float): Proportion of the data to be used as test set.
        random_state (int): Seed for random number generator.
        output_dir (str): Directory to save the preprocessed data.

    Returns:
        tuple: Training and test DataLoader objects.
    """
    # Download NLTK resources (stopwords and WordNet for lemmatization)
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load the dataset
    data = pd.read_csv(input_path)

    # Preprocess the lyrics
    data = preprocess_lyrics(data)
    print("Stopwords removed and text lemmatized.")

    # Encode the genre labels
    data, label_encoder = encode_labels(data)
    print("Genre Labels encoded.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(data, test_size=test_size, random_state=random_state)
    print("Dataset spitted.")

    # Prepare datasets for BERT
    train_dataset = LyricsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = LyricsDataset(X_test.tolist(), y_test.tolist(), tokenizer)
    print("Datasets prepared for BERT.")

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Data loaders prepared.")

    # Save the preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

    # Save the data loaders
    save_dataloaders(train_loader, test_loader, output_dir)

    return train_loader, test_loader



if __name__ == "__main__":
    input_path = r'results\cleaning\cleaned_data_20241224_112608.csv'

    # Get the data loaders
    train_loader, test_loader = preprocessing(input_path)

    
