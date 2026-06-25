# Use an official, lightweight Python runtime
FROM python:3.11-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 🔑 ADD THESE ENVIRONMENT VARIABLES TO CONTROL TENSORFLOW
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

# Install core system utilities that MediaPipe depends on
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set default port variable for Cloud Run
ENV PORT 8080
EXPOSE $PORT

# Cloud Run injects the PORT environment variable automatically
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 wsgi:app