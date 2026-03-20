# Docker Deployment Guide

## Overview
This guide explains how to deploy PalmPay using Docker to fix the libxcb.so.1 dependency issue that was occurring in the containerized environment.

## Issue Fixed
The original deployment was failing with:
```
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```

This occurred because OpenCV (cv2) requires several X11 and graphics libraries that weren't installed in the container. The Aptfile specified these dependencies but wasn't being properly processed by the deployment platform.

## Solution
Created a proper Dockerfile that:
1. Uses `python:3.11-slim` as the base image
2. Installs all required system dependencies for OpenCV and X11 libraries
3. Installs Python dependencies from requirements.txt
4. Properly exposes port 8080
5. Runs gunicorn with appropriate settings

## Local Testing

### Prerequisites
- Docker installed
- docker-compose installed

### Steps

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or manually build and run:**
   ```bash
   # Build the image
   docker build -t palmpay:latest .
   
   # Run the container
   docker run -p 8080:8080 \
     -e PORT=8080 \
     -e PYTHONUNBUFFERED=1 \
     palmpay:latest
   ```

3. **Access the application:**
   - Open your browser and go to `http://localhost:8080`

## Deployment to Production

### Deploying to Railway.app
1. Ensure you have a Railway.app account
2. Connect your GitHub repository to Railway
3. Railway will automatically detect the Dockerfile and use it for deployment
4. Set environment variables in Railway dashboard as needed

### Deploying to Heroku
If still using Heroku, create a `heroku.yml` file:
```yaml
build:
  docker:
    web: Dockerfile
run:
  web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

### Using Docker Hub
1. Build the image:
   ```bash
   docker build -t yourusername/palmpay:latest .
   ```

2. Push to Docker Hub:
   ```bash
   docker push yourusername/palmpay:latest
   ```

3. Deploy using docker run or docker-compose

## System Dependencies Included
- `libgl1-mesa-glx` - OpenGL graphics library
- `libglib2.0-0` - GLib library
- `libxcb1` - X11 window system library (the missing library)
- `libxkbcommon0` - X keyboard common library
- `libxkbcommon-x11-0` - X keyboard common X11 library
- `libxrender1` - X Render extension library
- `libxext6` - X11 extensions library
- `libx11-6` - X11 library
- `libxau6` - X authentication library
- `libxdmcp6` - X Display Manager Control Protocol library
- `libfreetype6` - Font rendering library
- `libfontconfig1` - Font configuration library
- `libharfbuzz0b` - Text shaping library

## Troubleshooting

### If you still see libxcb.so.1 errors:
1. Verify all system libraries are installed in the Dockerfile
2. Check that requirements.txt has `opencv-python-headless` (not `opencv-python`)
3. Ensure your deployment platform is using the Dockerfile

### If the container fails to start:
1. Check logs: `docker logs <container-id>`
2. Verify all Python packages are compatible with Python 3.11
3. Check that all model files (.h5) are included in the deployment

## Files Added/Modified
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Local testing configuration
- `.dockerignore` - Excludes unnecessary files from Docker build
- `requirements.txt` - Already properly configured with opencv-python-headless

## Next Steps
1. Test locally with docker-compose
2. Push changes to GitHub
3. Redeploy on your hosting platform
