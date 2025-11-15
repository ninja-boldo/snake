import { WebSocketServer } from 'ws';

const PORT = 3030;
const wss = new WebSocketServer({ port: PORT });

let gameClient = null;
let pythonClient = null;

// Message type identifiers
const MSG_TYPES = {
    PING_FRONTEND: 'pingfromfontend',
    PING_BACKEND: 'pingfrombackend',
    OBSERVATION: 'observation_space',
    REWARD: 'reward',
    DONE: 'done',
    ACTION: 'action',
    RESET: 'reset'
};

wss.on('connection', (ws, req) => {
    const clientIp = req.socket.remoteAddress;
    console.log(`\n🔌 New client connected from ${clientIp}`);
    console.log(`📊 Total clients: ${wss.clients.size}`);
    
    ws.on('message', (message) => {
        const msgStr = message.toString();
        const msgPreview = msgStr.length > 100 ? msgStr.substring(0, 100) + '...' : msgStr;
        
        // Identify and register clients
        if (msgStr.startsWith(MSG_TYPES.PING_FRONTEND)) {
            gameClient = ws;
            console.log('✅ Game client registered');
            ws.send('ack_frontend');
            return;
        }
        
        if (msgStr.startsWith(MSG_TYPES.PING_BACKEND)) {
            pythonClient = ws;
            console.log('✅ Python client registered');
            ws.send('ack_backend');
            return;
        }
        
        // Route messages from game to Python
        if (msgStr.startsWith(MSG_TYPES.OBSERVATION) || 
            msgStr.startsWith(MSG_TYPES.REWARD) ||
            msgStr.startsWith(MSG_TYPES.DONE)) {
            
            if (!gameClient) {
                gameClient = ws;
                console.log('✅ Game client auto-registered');
            }
            
            if (pythonClient && pythonClient.readyState === 1) {
                const msgType = msgStr.startsWith(MSG_TYPES.OBSERVATION) ? 'observation' :
                               msgStr.startsWith(MSG_TYPES.REWARD) ? 'reward' : 'done';
                console.log(`🎮→🐍 Forwarding ${msgType} to Python`);
                pythonClient.send(message);
            } else {
                console.log('⚠️  Python client not ready to receive messages');
            }
            return;
        }
        
        // Route messages from Python to game
        if (msgStr.startsWith(MSG_TYPES.ACTION)) {
            if (!pythonClient) {
                pythonClient = ws;
                console.log('✅ Python client auto-registered');
            }
            
            if (gameClient && gameClient.readyState === 1) {
                const actionNum = msgStr.replace(MSG_TYPES.ACTION, '');
                console.log(`🐍→🎮 Forwarding action ${actionNum} to game`);
                gameClient.send(message);
            } else {
                console.log('⚠️  Game client not ready to receive actions');
            }
            return;
        }
        
        // Route reset command from Python to game
        if (msgStr.includes(MSG_TYPES.RESET)) {
            if (!pythonClient) {
                pythonClient = ws;
                console.log('✅ Python client auto-registered (via reset)');
            }
            
            if (gameClient && gameClient.readyState === 1) {
                console.log('🐍→🎮 Forwarding reset command to game');
                gameClient.send('reset');
            } else {
                console.log('⚠️  Game client not ready for reset');
            }
            return;
        }
        
        // Unknown message type
        console.log(`⚠️  Unknown message type: ${msgPreview}`);
    });
    
    ws.on('close', () => {
        console.log('\n🔌 Client disconnected');
        
        if (ws === gameClient) {
            gameClient = null;
            console.log('❌ Game client disconnected');
        }
        
        if (ws === pythonClient) {
            pythonClient = null;
            console.log('❌ Python client disconnected');
        }
        
        console.log(`📊 Remaining clients: ${wss.clients.size}`);
    });
    
    ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error.message);
    });
});

wss.on('listening', () => {
    console.log('\n' + '='.repeat(60));
    console.log('🚀 WebSocket Server Started');
    console.log('='.repeat(60));
    console.log(`📡 Listening on ws://localhost:${PORT}`);
    console.log('⏳ Waiting for clients to connect...');
    console.log('   - Game client (browser)');
    console.log('   - Python client (training)');
    console.log('='.repeat(60) + '\n');
});

wss.on('error', (error) => {
    console.error('❌ Server error:', error.message);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\n🛑 Shutting down server...');
    wss.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});

// Health check
setInterval(() => {
    const status = {
        game: gameClient ? '✅ Connected' : '❌ Disconnected',
        python: pythonClient ? '✅ Connected' : '❌ Disconnected'
    };
    
    if (!gameClient || !pythonClient) {
        console.log(`\n📊 Status Check - Game: ${status.game} | Python: ${status.python}`);
    }
}, 30000); // Check every 30 seconds