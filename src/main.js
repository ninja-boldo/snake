const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

canvas.width = 800
canvas.height = 800

const width = canvas.width;
const height = canvas.height;

const blocks_row = 20

const imageData = ctx.createImageData(width, height);
const pixels = imageData.data; // Uint8ClampedArray

function setPixel(x, y, r, g, b, a = 255) {
    const index = (y * width + x) * 4;
    pixels[index]     = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
    pixels[index + 3] = a;
}

const setPixelRed = (x, y) => {
  setPixel(x, y, 255, 0, 0)
}

const setPixelGreen = (x, y) => {
  setPixel(x, y, 0, 255, 0)
}

const setPixelBlack = (x, y) => {
  setPixel(x, y, 0, 0, 0)
}


const range = (start, end) => {
  return Array.from({ length: end - start }, (_, i) => i + start);
}


let block_dim = width / blocks_row

const grid = [];
let count = 0

for (let y = 0; y < width; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {

        row.push(0);
    }
    grid.push(row);
}

const block_iter = []

let idx = 0
while(idx < width){
  block_iter.push(idx)
  idx += block_dim
}

const block_array = []

for (let y = 0; y < blocks_row; y++) {
    const row = [];
    for (let x = 0; x < blocks_row; x++) {

        row.push(0);
    }
    block_array.push(row);
}

const setBlock = (x, y) =>{
    for(idx_pixel_x in range(idx_block, idx_block + block_dim)){
      for(idx_pixel_y in range(idx_block, idx_block + block_dim)){
        setPixelGreen(idx_pixel_x, idx_pixel_y)
    }
  } 


for(let idx_x = 0; idx_x < width; i++){
  for(let idx_y = 0; idx_y < height; i++){
    
      count[idx_x][idx_y]

  }
}

ctx.putImageData(imageData, 0, 0);
