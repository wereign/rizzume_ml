FROM python:3.10-slim

# Set working directory
WORKDIR /usr/local/app

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

RUN  pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "--app-dir", "./inference", "main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]

