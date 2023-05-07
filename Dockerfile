# Use the official Python image as the parent image
# FROM python:3.9
FROM python:3.9-slim-buster
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt
# Install PyTorch packages with the custom index URL
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Expose the port that Flask is running on
EXPOSE 5000

# Run the command to start Flask
CMD ["python", "app.py"]
