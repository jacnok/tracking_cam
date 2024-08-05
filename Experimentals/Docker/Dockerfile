# Use the official Python image from the Python 3.11 series
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=100 Flask opencv-python-headless numpy pynput torch Pillow

# Expose ports
EXPOSE 5000 1259

# Set environment variables
ENV FLASK_APP=person_finder.py

# Run the application
CMD ["python", "person_finder.py"]
