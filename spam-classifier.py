# Install gdown to download dataset
!pip install -q gdown

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download and read dataset from Google Drive
file_id = "1xjNYjKTqAldnKAZwxaZD8RSbeJhlpJ1I"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "spam.csv", quiet=False)

# Load data
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Interactive prediction
print("\n SMS Spam Classifier is Ready!")
print("Type your message below (or type 'exit' to quit):\n")

while True:
    user_input = input("Your Message: ")
    if user_input.lower() == "exit":
        print("🔚 Exiting classifier.")
        break

    if len(user_input.strip()) < 4:
        print("⚠️ Message too short to classify meaningfully.\n")
        continue

    # Transform and predict with probability
    user_vec = vectorizer.transform([user_input])
    prob_spam = model.predict_proba(user_vec)[0][1]  # Probability of spam

    # Threshold to adjust sensitivity
    if prob_spam > 0.6:
        print(f" Prediction: SPAM (Confidence: {prob_spam:.2f})\n")
    else:
        print(f" Prediction: NOT SPAM (Confidence: {1 - prob_spam:.2f})\n")
