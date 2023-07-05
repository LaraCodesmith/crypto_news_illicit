import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm

# Loading the model
rf_model = joblib.load('rf_model.pkl')

# Loading the data
data_prepared = pd.read_csv('path/dataset_for model.csv') # Preprocessed text data
new_data = pd.read_csv('path/news_dataset.csv') # Original dataset where results will be added

# Transformation of dataset
vectorizer = TfidfVectorizer()
X_new = vectorizer.transform(data_prepared['text'])

with tqdm(total=len(data_prepared), desc="Predicting") as pbar:
    predictions = rf_model.predict(X_new)
    pbar.update(len(data_prepared))

# Adding predictions to the dataset as a new column
new_data['predicted_labels'] = predictions

# Saving the results
new_data.to_csv('path/news_predictions.csv', index=False)