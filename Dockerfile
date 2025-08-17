# Use a standard Python 3.11 image
FROM python:3.11

# Install curl, a tool for downloading files
RUN apt-get update && apt-get install -y curl

# Set the working directory in the container
WORKDIR /app

# Set the NLTK_DATA environment variable to a folder inside our app
ENV NLTK_DATA=/app/nltk_data

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK data to the specified directory
RUN python -m nltk.downloader -d $NLTK_DATA stopwords wordnet

# Download the model file from your GitHub Release
RUN curl -L -o sentiment_model_pipeline.pkl "https://github.com/Shelbyayush/Movie-Sentiment-Analysis/releases/download/v1.0/sentiment_model_pipeline.pkl"

# Copy the rest of your application code
COPY . .

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]