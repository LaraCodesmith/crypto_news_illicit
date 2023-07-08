import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

datasets = [
    {
        'name': 'cryptotrain',
        'path': 'path/crypto_news_parsed_2013-2017_train.csv',
        'output_path': 'path/cryptotrain_cleaned.csv',
        'selected_columns': ['title', 'text']
    },
    {
        'name': 'cryptoval',
        'path': 'path/crypto_news_parsed_2018_validation.csv',
        'output_path': 'path/cryptoval_cleaned.csv',
        'selected_columns': ['title', 'text']
    },
    {
        'name': 'cryptonews',
        'path': 'path/cryptonews.csv',
        'output_path': 'path/cryptonews_cleaned.csv',
        'selected_columns': ['title', 'text']
    }
]

final_dataset = pd.DataFrame()

def insert_spaces(text):
    pattern = r'(\b\w{5,})([A-Z])'
    result = re.sub(pattern, r'\1 \2', text)
    return result

for dataset in datasets:
    data = pd.read_csv(dataset['path'])

    # Select only the specified columns
    data = data[dataset['selected_columns']]

    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values in {dataset['name']} dataset:\n{missing_values}\n")

    # Delete rows with missing values
    data = data.dropna()

    # Insert spaces between some words where it is needed
    data = data.applymap(insert_spaces)

    # Clean the dataset
    data = data.apply(lambda x: x.str.lower())  # Convert to lowercase
    data = data.apply(lambda x: x.str.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation and quotation marks

    # Save the cleaned dataset
    data.to_csv(dataset['output_path'], index=False)

    # Concatenate the datasets
    final_dataset = pd.concat([final_dataset, data])

# Add labels
def add_labels(data, features):
    def is_connected(row):
        return int(
            any(feature in str(row.get('text', '')) or feature in str(row.get('title', '')) for feature in features))

    data['label'] = data.apply(is_connected, axis=1)
    return data

with open('path/keywords.txt', 'r') as file:
    keywords = [line.strip() for line in file]

labeled_data = add_labels(final_dataset, keywords)

# Overwrite first column with concatenated values
final_dataset['text'] = final_dataset['title'].astype(str) + ' ' + final_dataset['text'].astype(str)

# Drop the second column
final_dataset.drop('title', axis=1, inplace=True)

# Tokenization
final_dataset['text'] = final_dataset['text'].apply(lambda text: word_tokenize(text))

# Removing stop-words
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    tokens_without_stopwords = [token for token in tokens if token not in stop_words]
    return tokens_without_stopwords

final_dataset['text'] = final_dataset['text'].apply(remove_stopwords)

# Lemmatization
lemmatizer = WordNetLemmatizer()
final_dataset['text'] = final_dataset['text'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

print(final_dataset.shape)

# Save the final dataset
final_dataset.to_csv('path/dataset_for model.csv', index=False)