# 1. Start with a slim, lightweight Python base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Install system-level audio libraries required by librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first for Docker layer caching
COPY requirements.txt .

# 5. Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all your project files into the container (respecting .dockerignore)
COPY . .

# 7. Tell Python where the root directory is so it can find the /src folder
ENV PYTHONPATH="/code"

# 8. Grant permissions for Hugging Face's container system
RUN chmod -R 777 /code

# 9. Expose the port Hugging Face expects
EXPOSE 7860

# 10. Start the Streamlit frontend from your app folder
CMD ["streamlit", "run", "app/ui.py", "--server.port=7860", "--server.address=0.0.0.0"]