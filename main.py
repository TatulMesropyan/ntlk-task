import os
import pandas as pd
import re
from nltk.corpus import stopwords


def load_dataset(file_path, encoding='utf-8'):
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1')


def extract_hashtags(text):
    return re.findall(r'#\w+', text)


def extract_emails(text):
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)


def remove_usernames(text):
    return re.sub(r'@\w+', '', text)


def remove_links(text):
    return re.sub(r'http\S+', '', text)


def remove_non_ascii_symbols(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)


def to_lower(text):
    return text.lower()


stop_words = set(stopwords.words('english'))


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])


def remove_digits(text):
    return re.sub(r'\d+', '', text)


def remove_special_characters(text):
    special_characters = r'.,!?"#$%&*()+=-_[]{};:/\'"|<>\`~'

    return re.sub(f'[{re.escape(special_characters)}]', '', text)


def clean_text(text):
    text = remove_usernames(text)
    text = remove_links(text)
    text = remove_non_ascii_symbols(text)
    text = to_lower(text)
    text = remove_stop_words(text)
    text = remove_digits(text)
    text = remove_special_characters(text)
    return text


def find_csv_file(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv') and 'processed_' not in file_name:
            return os.path.join(directory, file_name)
    return None


def process_dataset(file_path):
    df = load_dataset(file_path)

    df['hashtags'] = df['text'].apply(extract_hashtags)
    df['emails'] = df['text'].apply(extract_emails)

    df['cleaned_text'] = df['text'].apply(clean_text)

    print(df[['text', 'cleaned_text', 'hashtags', 'emails']].head().to_string(index=False))

    return df


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = find_csv_file(current_directory)

    if csv_file_path is None:
        print("No unprocessed CSV files found in the current directory.")
    else:
        print(f"Processing file: {csv_file_path}")
        processed_df = process_dataset(csv_file_path)

        output_file = 'processed_' + os.path.basename(csv_file_path)
        processed_df.to_csv(output_file, index=False)
