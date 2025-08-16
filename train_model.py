import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving the model
import nltk
import os

print("pandas", pd.__version__)
print("nltk", nltk.__version__)
print("joblib", joblib.__version__)

# Download NLTK Data 
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK data (stopwords, wordnet)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data downloaded.")


# Load Data 
print("Loading data...")
try:
    df = pd.read_csv('C:\\Users\\Isha\\IdeaProjects\\untitled\\Python\\Projects\\Movie-Sentiment-Analysis\\IMDB Dataset.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same folder as this script.")
    exit()

#Text Preprocessing 
print("Setting up preprocessing tools...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Cleans and prepares the text for modeling.

    # Remove HTML tag
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and process words
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

print("Preprocessing text data... (This may take a few minutes for 50k reviews)")
# Applying preprocessing func to review column
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing complete.")

# Mapping sentiment labels to binary format: positive=1, negative=0 for modeling
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['cleaned_review']
y = df['sentiment']

# stratify=y to ensure the proportion of positive/negative reviews is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# Building and Training the Model using Pipeline 
print("Building and training the model pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)), # Convert text to TF-IDF features (top 5000)
    ('logreg', LogisticRegression(max_iter=1000, random_state=42)) # Logistic Regression classifier
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the Model
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print a detailed classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save the trained pipeline to a file for later use
# Saving the Model
output_dir = 'C:\\Users\\Isha\\IdeaProjects\\untitled\\Python\\Projects\\Movie-Sentiment-Analysis'
file_name = 'sentiment_model_pipeline.pkl'

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Construct full path to the file
file_path = os.path.join(output_dir, file_name)

# Save the trained pipeline to the path
joblib.dump(pipeline, file_path)
print(f"\nModel pipeline saved to '{file_path}'")

# Example Prediction on new data
def predict_sentiment(review, model_path='sentiment_model_pipeline.pkl'):
    """Loads the saved pipeline and predicts the sentiment of a single review."""
    # Load the pipeline from file
    loaded_pipeline = joblib.load(model_path)
    # Preprocess the new review w/ same steps
    cleaned_review = preprocess_text(review)
    # Predict the sentiment
    prediction = loaded_pipeline.predict([cleaned_review])
    # Return the human-readable label
    return "Positive" if prediction[0] == 1 else "Negative"

print("\n--- Testing with new reviews using the saved model ---")
review1 = "This movie was absolutely fantastic! The acting was brilliant and the storyline was gripping."
print(f"Review: '{review1}'\nPredicted Sentiment: {predict_sentiment(review1)}\n")

review2 = "A complete waste of time. The plot was predictable and the acting was terrible."
print(f"Review: '{review2}'\nPredicted Sentiment: {predict_sentiment(review2)}")
