from flask import Flask, request, render_template, jsonify
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize the Flask app
app = Flask(__name__)

# Ensuring NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and prepares text for sentiment prediction.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Loading the Trained Model
# Loading the sentiment analysis pipeline
try:
    model_pipeline = joblib.load('C:\\Users\\Isha\\IdeaProjects\\untitled\\Python\\Projects\\Movie-Sentiment-Analysis\\sentiment_model_pipeline.pkl')
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: 'sentiment_model_pipeline.pkl' not found.")
    print("Please run train_model.py first to create the model file.")
    model_pipeline = None # Set to None if loading fails

# Define Web Routes

@app.route('/')
def home():
    """
    Renders the main page of the web application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a review from the web form, predicts sentiment, and returns it.
    """
    if not model_pipeline:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    # Get the review text from the POST request form
    review_text = request.form['review']
    
    # Preprocess the input text
    cleaned_review = preprocess_text(review_text)
    
    # Predict the sentiment using the loaded pipeline
    prediction = model_pipeline.predict([cleaned_review])
    
    # Convert prediction (0 or 1) to a human-readable sentiment
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    # Return the result as JSON
    return jsonify({'sentiment': sentiment})

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)

