# Use official Python base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app files
COPY . .

# Expose port
EXPOSE 10000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
