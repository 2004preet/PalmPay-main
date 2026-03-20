FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and X11 libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libfreetype6 \
    libfontconfig1 \
    libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run gunicorn
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
