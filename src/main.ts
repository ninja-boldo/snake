const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
const canvas_obj = canvas.getContext("2d");

const websocket = new WebSocket("ws://localhost:3030");
let action = 0;
let reward = 0;
let blockExecution = true;
let lostGame = false;

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
    console.warn("about to send this reward: " + reward)
    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(prefix + reward);
    }
};

// Wait for WebSocket to open before starting
websocket.addEventListener("open", () => {
    console.log("WebSocket connected");
    websocket.send("pingfromfontend");
});

websocket.addEventListener("message", async (event: MessageEvent) => {
    const msgStr = event.data instanceof Blob ? await event.data.text() : String(event.data);
    console.log("received this message:", msgStr);
    
    // Handle reset command
    if (msgStr === "reset") {
        console.log("Received reset command");
        location.reload();
        return;
    }
    
    // Handle action commands
    if (msgStr.startsWith(actionPrefix)) {
        const actionValue = msgStr.slice(actionPrefix.length);
        const parsedAction = Number(actionValue);
        console.log("received this action value: " + parsedAction);

        if (Number.isInteger(parsedAction) && [0, 1, 2, 3].includes(parsedAction)) {
            dir = parsedAction;
            blockExecution = false;
        } else {
            console.error("Invalid action received:", actionValue);
        }
    }
});

websocket.addEventListener("error", (error) => {
    console.error("WebSocket error:", error);
});

websocket.addEventListener("close", () => {
    console.log("WebSocket connection closed");
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

const main = async (
    startBlockCount: number = 3,
    headColor: number[] = [50, 100, 250],
    bodyColor: number[] = [50, 100, 250],
    powerUpColor: number[] = [255, 100, 250],
    distToBorder: number = 4,
    msBetweenFrames: number = 200
) => {
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
            console.warn("Initial snake out of bounds; adjust distToBorder or startBlockCount");
            return;
        }
        worldMap[p.y][p.x] = (i === 0) ? 2 : 1;
    }

    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
    
    await sleep(msBetweenFrames);
    
    // Send initial state AFTER rendering
    sendReward(0);
    sendWorld();
    
    console.log("Sent initial state, waiting for first action...");

    while (blockExecution) {
        await sleep(10);
    }

    while (true) {
        reward = 0.01;
        let atePowerUp = false;
        const head = snake[0];
        
        oldDir = dir;
        
        let [newX, newY] = modCoordToDir(head.x, head.y, dir);

        // Check for power-up
        if (newX === PowerUpX && newY === PowerUpY) {
            reward += 10;
            atePowerUp = true;
            newPowerUp();
        }

        // Boundary check - game over
        if (newX < 0 || newX >= blocks_row || newY < 0 || newY >= blocks_row) {
            reward += -1000;
            sendReward(reward);
            sendWorld();
            console.log("Game over: hit wall at", newX, newY);
            lostGame = true;
            break;
        }

        // Body collision check - penalize but don't end game
        if (worldMap[newY][newX] === 1) {
            reward += -0.1;
            console.log("Body collision detected at", newX, newY, "- reverting to old direction");
            
            // Revert to old direction and recalculate position
            dir = oldDir;
            [newX, newY] = modCoordToDir(head.x, head.y, dir);
            
            // If reverting still causes collision, we're stuck - end game
            if (newX < 0 || newX >= blocks_row || newY < 0 || newY >= blocks_row || worldMap[newY][newX] === 1) {
                reward += -10;
                console.log("Game over: stuck after body collision");
                lostGame = false;

            }
        }

        // Update world map and snake
        worldMap[head.y][head.x] = 1;
        snake.unshift({ x: newX, y: newY });
        worldMap[newY][newX] = 2;

        if (!atePowerUp) {
            const tail = snake.pop()!;
            worldMap[tail.y][tail.x] = 0;

            //incentivize going to the powerUp
            reward += 10/( (PowerUpX - head.x + PowerUpY - head.y)**2 )
        }



        // Send reward and observation once per step
        sendReward(reward);
        sendWorld();

        renderWorldMap(headColor, bodyColor, powerUpColor);
        canvas_obj.putImageData(imageData, 0, 0);

        await sleep(msBetweenFrames);
        
        blockExecution = true;
        while (blockExecution) {
            await sleep(10);
        }
    }

    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
    
    await sleep(1000);
    console.log("Restarting game...");

    for (let y = 0; y < blocks_row; y++) {
        for (let x = 0; x < blocks_row; x++) {
            worldMap[y][x] = 0;
        }
    }

    dir = 1;
    oldDir = 1;
    blockExecution = true;
    lostGame = false;
    
    await main(startBlockCount, headColor, bodyColor, powerUpColor, distToBorder, msBetweenFrames);
};

// Wait for WebSocket to be ready before starting the game
const waitForWebSocket = async () => {
    while (websocket.readyState !== WebSocket.OPEN) {
        await sleep(100);
    }
};

const frameDelay = 20;
const startBlocks = 3;
const snakeColor = [50, 200, 120];
const powerUpColor = [255, 200, 120];
const distToBorder = 4;

waitForWebSocket().then(() => {
    main(startBlocks, snakeColor, snakeColor, powerUpColor, distToBorder, frameDelay);
});