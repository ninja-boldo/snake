export {}; 

const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
const canvas_obj = canvas.getContext("2d");

let dir: number = 1
let oldDir: number = 1
let PowerUpX: number, PowerUpY: number

document.addEventListener('keydown', function(event) {
    if(event.key === "ArrowUp"){
        console.log("pressed up")
        oldDir = dir
        dir = 0
    }
    else if(event.key === "ArrowRight"){
        console.log("pressed right")
        oldDir = dir
        dir = 1
    }
    else if(event.key === "ArrowDown"){
        console.log("pressed down")
        oldDir = dir
        dir = 2
    }
    else if(event.key === "ArrowLeft"){
        console.log("pressed left")
        oldDir = dir
        dir = 3
    }    
});

if (!canvas_obj) {
    throw new Error("Could not get 2D context from canvas");
}

canvas.width = 800;
canvas.height = 800;

const width = canvas.width;
const height = canvas.height;

const blocks_row = 20;
const block_dim = Math.floor(width / blocks_row);

const worldMap: number[][] = []; // worldMap[y][x] (row, col)

for (let y = 0; y < blocks_row; y++) {
    const row: number[] = [];
    for (let x = 0; x < blocks_row; x++) {
        row.push(0);
    }
    worldMap.push(row);
}

const imageData = canvas_obj.createImageData(width, height);
const pixels = imageData.data;

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

const setPixelCustomColor = (x: number, y: number, color: number[]) => setPixel(x, y, color[0], color[1], color[2]);

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

// convert a logical (x,y) block coord by direction (0 up, 1 right, 2 down, 3 left)
const modCoordToDir = (x: number, y: number, dir: number): [number, number] => {
    if (dir === 0) return [x, y - 1];
    if (dir === 1) return [x + 1, y];
    if (dir === 2) return [x, y + 1];
    if (dir === 3) return [x - 1, y];
    throw new Error("invalid dir");
};

// Render worldMap where 3=PowerUp, 2=head, 1=body
const renderWorldMap = (headColor: number[], bodyColor: number[], powerUpColor: number[]) => {
    // Clear pixel buffer to black
    setWholeCanvas([0, 0, 0]);

    for (let y = 0; y < blocks_row; y++) {
        for (let x = 0; x < blocks_row; x++) {
            if (worldMap[y][x] === 3) {
                setBlockCustom(x, y, powerUpColor);
            }
            else if(worldMap[y][x] === 2){
                setBlockCustom(x, y, headColor);
            }
            else if (worldMap[y][x] === 1) {
                setBlockCustom(x, y, bodyColor);
            }
        }
    }
};

const newPowerUp = () => {
    let randRow = Math.round(Math.random() * (blocks_row - 1));
    let randCol = Math.round(Math.random() * (blocks_row - 1));

    while(worldMap[randCol][randRow] != 0){
        randRow = Math.round(Math.random() * (blocks_row - 1));
        randCol = Math.round(Math.random() * (blocks_row - 1));
    }
    worldMap[randCol][randRow] = 3
    PowerUpX = randRow
    PowerUpY = randCol
}

interface Point { x: number; y: number; }

const main = async (
    startBlockCount: number = 3,
    headColor: number[] = [50, 100, 250],
    bodyColor: number[] = [50, 100, 250],
    powerUpColor: number[] = [255, 100, 250],
    distToBorder: number = 4,
    msBetweenFrames: number = 200
) => {

    // random starting position within borders
    const randX = Math.floor(Math.random() * (blocks_row - 2 * distToBorder)) + distToBorder;
    const randY = Math.floor(Math.random() * (blocks_row - 2 * distToBorder)) + distToBorder;
    newPowerUp()

    // Build ordered snake array with head at index 0.
    // If dir === 1 (right), body should be left of the head.
    const snake: Point[] = [];
    for (let i = 0; i < startBlockCount; i++) {
        snake.push({ x: randX - i, y: randY }); // head at index 0
    }

    // fill worldMap with snake
    for (let i = 0; i < snake.length; i++) {
        const p = snake[i];
        if (p.x < 0 || p.x >= blocks_row || p.y < 0 || p.y >= blocks_row) {
            console.warn("Initial snake out of bounds; adjust distToBorder or startBlockCount");
            return;
        }
        worldMap[p.y][p.x] = (i === 0) ? 2 : 1;
    }

    // initial render
    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
    await sleep(msBetweenFrames);

    // simple loop: move with fixed direction; stops on collision/border
    while (true) {
        let atePowerUp = false
        const head = snake[0];
        let [newX, newY] = modCoordToDir(head.x, head.y, dir);

        if (newX == PowerUpX && newY == PowerUpY){
            atePowerUp = true
            newPowerUp()
        }
        // boundary / collision check
        if (newX < 0 || newX >= blocks_row || newY < 0 || newY >= blocks_row) {
            console.log("Game over: hit wall at", newX, newY);
            break;
        }
        if (worldMap[newY][newX] === 1) {
            console.log("hit body at", newX, newY);
            dir = oldDir;
            [newX, newY] = modCoordToDir(head.x, head.y, dir);
        }

        worldMap[head.y][head.x] = 1;

        snake.unshift({ x: newX, y: newY });
        worldMap[newY][newX] = 2;

        //remove tail if no PowerUp was eatten
        if(!atePowerUp){
            const tail = snake.pop()!;
            worldMap[tail.y][tail.x] = 0;
        }

        // render and wait
        renderWorldMap(headColor, bodyColor, powerUpColor);
        canvas_obj.putImageData(imageData, 0, 0);

        await sleep(msBetweenFrames);
    }

    // final render so you can see the final state
    renderWorldMap(headColor, bodyColor, powerUpColor);
    canvas_obj.putImageData(imageData, 0, 0);
};

const frameDelay = 200
const startBlocks = 3
const snakeColor = [50, 200, 120]
const powerUpColor = [255, 200, 120]
const distToBorder = 4

main(startBlocks, snakeColor,snakeColor, powerUpColor, distToBorder, frameDelay);