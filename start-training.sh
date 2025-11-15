#!/bin/bash
sleep 8
chromium --headless --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --remote-debugging-port=9222 http://localhost:5173 &
sleep 2
python rl/train.py