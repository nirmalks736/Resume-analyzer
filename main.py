import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import fitz  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
df = pd.read_csv('./archive (1)/Resume/Resume.csv')

# Text Cleaning Function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['Resume_str'] = df['Resume_str'].apply(clean_text)

# Encode Category Labels
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df['Resume_str'])
y = df['Category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
pickle.dump(lr, open('resume_classifier.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# Function to Predict Resume Category
def predict_category(resume_text):
    cleaned_resume = clean_text(resume_text)
    vectorized_resume = tfidf.transform([cleaned_resume])
    prediction = lr.predict(vectorized_resume)[0]
    return label_encoder.inverse_transform([prediction])[0]

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Function to Predict from PDF
def predict_from_pdf(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    return predict_category(extracted_text)

# Example Usage with PDF
print("Predicted Category:", predict_from_pdf("./Nirmal KS (1).pdf"))
