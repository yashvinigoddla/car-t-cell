# Gunicorn configuration file for Flask-SocketIO application
import os

# Bind to 0.0.0.0 with port from environment variable (Render sets PORT)
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Use eventlet worker class for WebSocket support with Flask-SocketIO
worker_class = "eventlet"

# Number of worker processes
# For eventlet, use 1 worker as it handles concurrency internally
workers = 1

# Timeout for worker processes (in seconds)
timeout = 120

# Enable access logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload the application to save memory
preload_app = True

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

