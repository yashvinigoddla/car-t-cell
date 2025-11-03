# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that may be needed for scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Demo directory contents first
COPY Demo/requirements.txt /app/requirements.txt
COPY Demo/app.py /app/app.py
COPY Demo/ppo_cart_1000000.zip /app/ppo_cart_1000000.zip
COPY Demo/gunicorn_config.py /app/gunicorn_config.py

# Copy static and template directories
COPY Demo/templates/ /app/templates/
COPY Demo/static/ /app/static/

# Copy parent directory dependencies (environment.py and simulation.py)
COPY environment.py /app/environment.py
COPY simulation.py /app/simulation.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Expose the port (Render will set the PORT env var)
EXPOSE ${PORT}

# Use gunicorn with eventlet worker for production
CMD gunicorn --config gunicorn_config.py app:app

