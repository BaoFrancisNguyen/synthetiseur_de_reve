FROM python:3.11

# Set work directory
WORKDIR /app

# Copy requirements if you have one, else skip this step
COPY requirements.txt .

# Install dependencies (if you have requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Default command (run your app)
CMD ["python", "app.py"]