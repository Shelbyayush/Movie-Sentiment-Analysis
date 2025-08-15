# Movie Review Sentiment Analysis

This project is a Natural Language Processing (NLP) application designed to perform sentiment analysis on movie reviews. Using the IMDB dataset of 50,000 reviews, the system preprocesses raw text data, converts it into numerical features using TF-IDF, and trains a Logistic Regression model to classify the sentiment. The final model is deployed in a lightweight Flask web application, allowing users to input their own reviews and receive instant predictions.

## Features
- **Text Preprocessing & Model Training**: Cleans and prepares raw text data for modeling and uses TF-IDF vectorization and a Logistic Regression classifier.
- **Web Interface**: A simple web app built with Flask to interact with the model.

## How to Run
1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the training script: `python train_model.py`
4. Run the web application: `python app.py`

## Author
- **[Ayush Chaudhary]** - 
Github: https://github.com/Shelbyayush
LinkedIn: https://www.linkedin.com/in/ayush-chaudhary-ba00b8248/