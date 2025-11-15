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

# Copy and set permissions for startup script
COPY start-training.sh /app/start-training.sh
RUN chmod +x /app/start-training.sh

# Expose ports
EXPOSE 3030 5173

# Install process manager
RUN npm install -g concurrently

CMD ["concurrently", \
     "--names", "WS,VITE,BROWSER", \
     "--prefix", "[{name}]", \
     "--prefix-colors", "blue,green,magenta", \
     "node src/websocket-server.js", \
     "sleep 5 && npm run dev -- --host 0.0.0.0", \
     "/app/start-training.sh"]