# Using aN official Python runtime as a parent image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app/backend

# Install system dependencies if any (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt # Optional: Not sure this is needed, but ganna put this here for now

# Install Python dependencies
# Consider using a virtual environment inside Docker if preferred, but often not necessary for simple images
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# If you have dev dependencies needed for any build steps or checks during build:
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the pyproject.toml and the source code for the installable package
COPY pyproject.toml pyproject.toml
COPY fastapi_build ./fastapi_build

# Install the local package (fastapi_build)
# This command assumes your pyproject.toml is set up for this
RUN pip install --no-cache-dir .


EXPOSE 8000

# Define the command to run the application
# CMD ["uvicorn", "fastapi_build.main:app", "--host", "0.0.0.0", "--port", "8000"]
# It's often better to use an entrypoint script for more complex startup logic or to handle `dumb-init`
# For now, CMD is fine. Ensure your GOOGLE_API_KEY and other env vars are passed at runtime.
CMD ["akd web"]
#CMD ["sh", "-c", "echo 'Starting Uvicorn...' && uvicorn fastapi_build.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
#test the FastAPI app first
