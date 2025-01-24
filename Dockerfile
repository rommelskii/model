# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR .

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY . .

# Expose the port your app will run on
EXPOSE 6969

WORKDIR /app

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:6969", "main:app"]


