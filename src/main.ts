const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
const canvas_obj = canvas.getContext("2d");

const websocket = new WebSocket("ws://localhost:3030");
let action = 0;
let reward = 0;
let blockExecution = true;
let gameOver = false;
let isResetting = false;

const actionPrefix = "action";

let dir: number = 1;
let oldDir: number = 1;
let PowerUpX: number, PowerUpY: number;

if (!canvas_obj) {
    throw new Error("Could not get 2D context from canvas");
}

canvas.width = 800;
canvas.height = 800;

const width = canvas.width;
const height = canvas.height;

const blocks_row = 20;
const block_dim = Math.floor(width / blocks_row);

const worldMap: number[][] = [];

for (let y = 0; y < blocks_row; y++) {
    const row: number[] = [];
    for (let x = 0; x < blocks_row; x++) {
        row.push(0);
    }
    worldMap.push(row);
}

const imageData = canvas_obj.createImageData(width, height);
const pixels = imageData.data;

const sendWorld = (prefix: string = "observation_space") => {
    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(prefix + JSON.stringify(worldMap));
    }
};

const sendReward = (reward: number, prefix: string = "reward") => {
    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(prefix + reward);
    }
};

// Wait for WebSocket to open before starting
websocket.addEventListener("open", () => {
    console.log("✓ WebSocket connected");
    websocket.send("pingfromfontend");
});

websocket.addEventListener("message", async (event: MessageEvent) => {
    const msgStr = event.data instanceof Blob ? await event.data.text() : String(event.data);
    
    // Ignore ping/ack messages
    if (msgStr.startsWith("ping") || msgStr.startsWith("ack")) {
        return;
    }
    
    // Handle reset command
    if (msgStr === "reset") {
        console.log("🔄 Received reset command");
        isResetting = true;
        gameOver = true;
        blockExecution = false;
        return;
    }
    
    // Handle action commands
    if (msgStr.startsWith(actionPrefix)) {
        const actionValue = msgStr.slice(actionPrefix.length);
        const parsedAction = Number(actionValue);

        if (Number.isInteger(parsedAction) && [0, 1, 2, 3].includes(parsedAction)) {
            const oppositeDir = (oldDir + 2) % 4;
            if (parsedAction === oppositeDir) {
                dir = oldDir;
                reward += -0.5;
            } else {
                dir = parsedAction;
            }
            blockExecution = false;
        }
    }
});

websocket.addEventListener("error", (error) => {
    console.error("❌ WebSocket error:", error);
});

websocket.addEventListener("close", () => {
    console.log("❌ WebSocket connection closed");
});

function setPixel(x: number, y: number, r: number, g: number, b: number, a: number = 255) {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    if (xi < 0 || xi >= width || yi < 0 || yi >= height) return;
    const index = (yi * width + xi) * 4;
    pixels[index]     = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
    pixels[index + 3] = a;
}

const setPixelCustomColor = (x: number, y: number, color: number[]) => 
    setPixel(x, y, color[0], color[1], color[2]);

function setBlockCustom(blockX: number, blockY: number, color: number[]) {
    const startX = blockX * block_dim;
    const startY = blockY * block_dim;
    const endX = startX + block_dim;
    const endY = startY + block_dim;
    for (let px = startX; px < endX; px++) {
        for (let py = startY; py < endY; py++) {
            setPixelCustomColor(px, py, color);
        }
    }
}

const setWholeCanvas = (color: number[]) => {
    for (let by = 0; by < blocks_row; by++) {
        for (let bx = 0; bx < blocks_row; bx++) {
            setBlockCustom(bx, by, color);
        }
    }
};

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const modCoordToDir = (x: number, y: number, dir: number): [number, number] => {
    if (dir === 0) return [x, y - 1];
    if (dir === 1) return [x + 1, y];
    if (dir === 2) return [x, y + 1];
    if (dir === 3) return [x - 1, y];
    throw new Error("invalid dir");
};

const renderWorldMap = (headColor: number[], bodyColor: number[], powerUpColor: number[]) => {
    setWholeCanvas([0, 0, 0]);

    for (let y = 0; y < blocks_row; y++) {
        for (let x = 0; x < blocks_row; x++) {
            if (worldMap[y][x] === 3) {
                setBlockCustom(x, y, powerUpColor);
            } else if (worldMap[y][x] === 2) {
                setBlockCustom(x, y, headColor);
            } else if (worldMap[y][x] === 1) {
                setBlockCustom(x, y, bodyColor);
            }
        }
    }
};

const newPowerUp = () => {
    let randRow = Math.round(Math.random() * (blocks_row - 1));
    let randCol = Math.round(Math.random() * (blocks_row - 1));

    while (worldMap[randCol][randRow] !== 0) {
        randRow = Math.round(Math.random() * (blocks_row - 1));
        randCol = Math.round(Math.random() * (blocks_row - 1));
    }
    worldMap[randCol][randRow] = 3;
    PowerUpX = randRow;
    PowerUpY = randCol;
};

interface Point { x: number; y: number; }

const resetGame = () => {
    //console.log("🔄 Resetting game state...");
    
    // Clear world map
    for (let y = 0; y < blocks_row; y++) {
        for (let x = 0; x < blocks_row; x++) {
            worldMap[y][x] = 0;
        }
    }
    
    // Reset game state
    dir = 1;
    oldDir = 1;
    blockExecution = true;
    gameOver = false;
    isResetting = false;
    
    // Clear canvas
    setWholeCanvas([0, 0, 0]);
    canvas_obj.putImageData(imageData, 0, 0);
    
    //console.log("✓ Game state reset complete");
};

const main = async (
    startBlockCount: number = 3,
    headColor: number[] = [50, 100, 250],
    bodyColor: number[] = [50, 100, 250],
    powerUpColor: number[] = [255, 100, 250],
    distToBorder: number = 4,
    msBetweenFrames: number = 5
) => {
    if (gameOver && !isResetting) {
        console.log("⚠️ Game already over, ignoring main() call");
        return;
    }
    
    //console.log("🎮 Initializing game...");
    
    const randX = Math.floor(Math.random() * (blocks_row - 2 * distToBorder)) + distToBorder;
    const randY = Math.floor(Math.random() * (blocks_row - 2 * distToBorder)) + distToBorder;
    newPowerUp();

    const snake: Point[] = [];
    for (let i = 0; i < startBlockCount; i++) {
        snake.push({ x: randX - i, y: randY });
    }

    for (let i = 0; i < snake.length; i++) {
        const p = snake[i];
        if (p.x < 0 || p.x >= blocks_row || p.y < 0 || p.y >= blocks_row) {
            console.warn("Initial snake out of bounds");
            return;
        }
        worldMap[p.y][p.x] = (i === 0) ? 2 : 1;
    }

    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
    
    // CRITICAL: Send initial state immediately without delay
    //console.log("📤 Sending initial state...");
    sendReward(0);
    sendWorld();
    
    // Small frame delay only if specified
    if (msBetweenFrames > 0) {
        await sleep(msBetweenFrames);
    }
    
    //console.log("⏳ Waiting for first action...");

    // Wait for first action
    while (blockExecution && !gameOver) {
        await sleep(5);  // Reduced from 10ms to 5ms for faster response
    }

    // Main game loop
    //console.log("▶️ Starting game loop");
    while (!gameOver) {
        reward = 0.1;
        let atePowerUp = false;
        const head = snake[0];
        
        oldDir = dir;
        
        let [newX, newY] = modCoordToDir(head.x, head.y, dir);

        // Boundary check
        if (newX < 0 || newX >= blocks_row || newY < 0 || newY >= blocks_row) {
            reward = -1000;
            gameOver = true;
            
            sendReward(reward);
            sendWorld();
            break;
        }

        // Body collision check
        if (worldMap[newY][newX] === 1) {
            reward = -800;
            gameOver = true;
            
            sendReward(reward);
            sendWorld();
            break;
        }

        // Check for power-up
        if (newX === PowerUpX && newY === PowerUpY) {
            reward += 30;
            atePowerUp = true;
            newPowerUp();
        }

        // Update world map and snake
        worldMap[head.y][head.x] = 1;
        snake.unshift({ x: newX, y: newY });
        worldMap[newY][newX] = 2;

        if (!atePowerUp) {
            const tail = snake.pop()!;
            worldMap[tail.y][tail.x] = 0;

            const dx = Math.abs(PowerUpX - newX);
            const dy = Math.abs(PowerUpY - newY);
            const distance = dx + dy;
            
            if (distance > 0) {
                reward += 0.2 / distance;
            } else {
                reward += 0.1;
            }
        }

        // Send reward and observation
        sendReward(reward);
        sendWorld();

        renderWorldMap(headColor, bodyColor, powerUpColor);
        canvas_obj.putImageData(imageData, 0, 0);

        if (msBetweenFrames > 0) {
            await sleep(msBetweenFrames);
        }
        
        // Wait for next action
        blockExecution = true;
        while (blockExecution && !gameOver) {
            await sleep(5);  // Reduced from 10ms to 5ms
        }
    }

    // Game over - render final state
    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
    
    //console.log("🏁 Game ended. Waiting for reset command...");
    
    // Wait for reset command
    while (!isResetting) {
        await sleep(50);
    }
    
    // Reset and immediately start new game - NO DELAY
    resetGame();
    
    // Recursively start new game immediately
    //console.log("🚀 Starting new episode immediately...");
    main(startBlockCount, headColor, bodyColor, powerUpColor, distToBorder, msBetweenFrames);
};

// Wait for WebSocket to be ready before starting the game
const waitForWebSocket = async () => {
    console.log("⏳ Waiting for WebSocket connection...");
    while (websocket.readyState !== WebSocket.OPEN) {
        await sleep(100);
    }
    console.log("✓ WebSocket ready");
};

// Configuration - adjust frameDelay to control game speed
// 0 = fastest, 100 = slower (good for watching), 20 = balanced
const frameDelay = 0;  // Change this to slow down visualization
const startBlocks = 3;
const snakeColor = [50, 200, 120];
const powerUpColor = [255, 200, 120];
const distToBorder = 4;

console.log("🐍 Snake RL Game Starting...");
console.log(`⚙️  Frame delay: ${frameDelay}ms`);
waitForWebSocket().then(() => {
    console.log("🚀 Launching initial game");
    main(startBlocks, snakeColor, snakeColor, powerUpColor, distToBorder, frameDelay);
});