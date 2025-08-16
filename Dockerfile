# Use a standard Python 3.11 image
FROM python:3.11

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs

# Set the working directory in the container
WORKDIR /app

# Copy the Git repository context first
COPY .git ./.git

# Copy the rest of your application code
COPY . .

# Initialize LFS and pull the large model file
RUN git lfs install && git lfs pull

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]