# Data Cleaning and Extraction for Coachella Tweets Dataset

## Task
The task is to perform data cleaning and extraction on the Coachella Tweets dataset, which contains tweets related to the Coachella music festival. The following steps are required:
- Find all the hashtags in the tweets and save them in a separate column named "hashtags".
- Find all the emails in the tweets and save them in a separate column named "emails".
- Write functions to remove unwanted elements from the tweets and apply them to the new "cleaned_text" column:
  - `remove_usernames`: This function removes usernames starting with '@'.
  - `remove_links`: This function removes links starting with 'http' or 'https'.
  - `remove_non_ascii_symbols`: This function removes symbols not part of the ASCII character set.
  - `to_lower`: This function converts text to lower case.
  - `remove_stop_words`: This function removes common words that do not add much meaning.
  - `remove_digits`: This function removes numbers.
  - `remove_special_characters`: This function removes punctuation marks and other special characters.

## Solution
```python
import os
import pandas as pd
import re
from nltk.corpus import stopwords

# Function to load the dataset from a CSV file
def load_dataset(file_path, encoding='utf-8'):
    # Attempt to read the CSV file with the specified encoding
    try:
        return pd.read_csv(file_path, encoding=encoding)
    # Handle UnicodeDecodeError by trying a different encoding
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1')

# Function to extract hashtags from text
def extract_hashtags(text):
    return re.findall(r'#\w+', text)

# Function to extract emails from text
def extract_emails(text):
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

# Function to remove usernames starting with '@'
def remove_usernames(text):
    return re.sub(r'@\w+', '', text)

# Function to remove links starting with 'http' or 'https'
def remove_links(text):
    return re.sub(r'http\S+', '', text)

# Function to remove symbols not part of the ASCII character set
def remove_non_ascii_symbols(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to convert text to lower case
def to_lower(text):
    return text.lower()

# Set of English stopwords for removing common words
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from text
def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Function to remove digits from text
def remove_digits(text):
    return re.sub(r'\d+', '', text)

# Function to remove punctuation marks and special characters from text
def remove_special_characters(text):
    # Define the set of punctuation marks and special characters to remove
    special_characters = r'.,!?"#$%&*()+=-_[]{};:/\'"|<>\`~'
    # Remove the specified characters from the text
    return re.sub(f'[{re.escape(special_characters)}]', '', text)

# Function to clean the tweet text by applying all cleaning functions
def clean_text(text):
    text = remove_usernames(text)
    text = remove_links(text)
    text = remove_non_ascii_symbols(text)
    text = to_lower(text)
    text = remove_stop_words(text)
    text = remove_digits(text)
    text = remove_special_characters(text)
    return text

# Function to find the first CSV file in the current directory
def find_csv_file(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            return os.path.join(directory, file_name)
    return None

# Function to process the dataset
def process_dataset(file_path):
    # Load the dataset from the CSV file
    df = load_dataset(file_path)
    # Extract hashtags and emails from the tweet text
    df['hashtags'] = df['text'].apply(extract_hashtags)
    df['emails'] = df['text'].apply(extract_emails)
    # Clean the tweet text and create a new column 'cleaned_text'
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Display the processed data as a table
    print(df[['text', 'cleaned_text', 'hashtags', 'emails']].head().to_string(index=False))
    return df

# Main script execution
if __name__ == "__main__":
    # Find the first CSV file in the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = find_csv_file(current_directory)

    # If no CSV file is found, print a message
    if csv_file_path is None:
        print("No CSV files found in the current directory.")
    else:
        # Process the dataset and save the processed data to a new CSV file
        print(f"Processing file: {csv_file_path}")
        processed_df = process_dataset(csv_file_path)
        output_file = 'processed_' + os.path.basename(csv_file_path)
        processed_df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
