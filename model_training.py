import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Reading a data for model building
data = pd.read_csv('path/dataset_for model.csv')

# Splitting the data into training and testing sets
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training data fitting and transformation
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train['text'])

# Test data transformation
X_test_tfidf = vectorizer.transform(X_test['text'])

# Checking a dimensions of training data
print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"y_train shape: {y_train.shape}")

# Model training
rf_model = RandomForestClassifier(verbose=2, n_jobs=-1, n_estimators=500)

rf_model.fit(X_train_tfidf, y_train)

# Saving the model
joblib.dump(rf_model, 'rf_model.pkl')

# Comparing predicted labels with the actual labels and summarize the results
y_pred = rf_model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print(report)
