FROM python:3.11-alpine

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of application code
COPY . .

# Run git lfs pull to download the model
RUN git lfs pull

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]