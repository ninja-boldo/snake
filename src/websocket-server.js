import { WebSocketServer } from 'ws';

const PORT = 3030;
const wss = new WebSocketServer({ port: PORT });

let gameClient = null;
let pythonClient = null;

wss.on('connection', (ws, req) => {
    const clientIp = req.socket.remoteAddress;
    console.log(`New client connected from ${clientIp}`);
    console.log(`Total clients: ${wss.clients.size}`);
    
    ws.on('message', (message) => {
        const msgStr = message.toString();
        console.log(`Received (${msgStr.length} chars):`, msgStr.substring(0, 100));
        
        // Identify client type based on first message
        if (msgStr.startsWith('pingfromfontend')){
            if (!gameClient) {
                gameClient = ws;
                console.log('✓ Game client registered');
            }
        }
        if (msgStr.startsWith('pingfrombackend')){
            if (!pythonClient) {
                pythonClient = ws;
                console.log('✓ Python client registered');
            }
        }
        if (msgStr.startsWith('observation_space') || msgStr.startsWith('reward')) {
            // Message from game -> forward to Python
            if (!gameClient) {
                gameClient = ws;
                console.log('✓ Game client registered');
            }
            if (pythonClient && pythonClient.readyState === 1) {
                console.log('→ Forwarding to Python client');
                pythonClient.send(message);
            } else {
                console.log('✗ Python client not ready');
            }
        } else if (msgStr.startsWith('action')) {
            // Message from Python -> forward to game
            if (!pythonClient) {
                pythonClient = ws;
                console.log('✓ Python client registered');
            }
            if (gameClient) {
                console.log('→ Forwarding action to game client');
                gameClient.send(message);
            } else {
                console.log('✗ Game client not ready');
            }
        } else if (msgStr.includes('reset') || msgStr.includes('command')) {
            // Reset command from Python -> forward to game
            if (!pythonClient) {
                pythonClient = ws;
                console.log('✓ Python client registered (via reset)');
            }
            if (gameClient) {
                console.log('→ Forwarding reset to game client');
                gameClient.send('reset');
            } else {
                console.log('✗ Game client not ready for reset');
            }
        } else {
            console.log(`⚠ Unknown message type: ${msgStr.substring(0, 50)}`);
        }
    });
    
    ws.on('close', () => {
        console.log('Client disconnected');
        if (ws === gameClient) {
            gameClient = null;
            console.log('✗ Game client disconnected');
        }
        if (ws === pythonClient) {
            pythonClient = null;
            console.log('✗ Python client disconnected');
        }
        console.log(`Remaining clients: ${wss.clients.size}`);
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
    
    // Send a ping to verify connection
    ws.send('ping');
});

wss.on('listening', () => {
    console.log(`🚀 WebSocket server running on ws://localhost:${PORT}`);
    console.log('Waiting for clients...');
});

wss.on('error', (error) => {
    console.error('Server error:', error);
});