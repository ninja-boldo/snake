FROM nikolaik/python-nodejs:python3.11-nodejs20

WORKDIR /app

# Install Chromium for headless browser
RUN apt-get update && \
    apt-get install -y chromium chromium-driver && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY package*.json requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir uv selenium && \
    uv pip install --no-cache-dir -r requirements.txt --system && \
    npm install

# Copy application code
COPY . .

CMD python rl/train.py
