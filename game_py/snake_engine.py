import inspect
import random
import numpy as np
from typing import Tuple, List

#declare beginning variables
dimensions = 10
distToBorder = 3
startBlocks = 3
worldMap = np.zeros((dimensions, dimensions))
addXBlocks = 1

lostGame = False
borderCollsion = False
bodyCollision = False

dir = 1
atePowerUp = False
errorInLine = -1
reward = 0



class BodyElement:
    def __init__(self, x, y, isHead: bool=False) -> None:
        self.x = x
        self.y = y
        self.isHead = isHead
        
class PowerUp:
    def __init__(self, x, y, addBlocks=1) -> None:
        self.x = x
        self.y = y
        self.addBlocks = addBlocks
class Coordinates:
    def __init__(self, x, y) -> None:
        self.x = x 
        self.y = y
        
bodyElements: List[BodyElement] = []
powerUps: List[PowerUp] = []

def line_of_call():
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return -1
    return frame.f_back.f_lineno

def wipeWorldMapInplace():
    global worldMap
    worldMap = np.zeros((dimensions, dimensions))
    
def wipeWorldMap() -> np.ndarray:
    return np.zeros((dimensions, dimensions))
    
def chooseDir(disallowed: list[int]) -> int:
    #0 is up, 1 is right, 2 is down and 3 is left
    directions = [0, 1, 2, 3]
    for disallow in disallowed:
        directions.remove(disallow)
    if len(directions) == 0:
        raise Exception("you to allow at least one direction")
    if len(directions) == 1:
        return directions[0]
    
    return random.choice(directions)

def addBlock(x: int, y: int, elementType: int) -> None:
    #elementype = 1 is a normal body, 2 is the head of the snake, 0 is void, 3 is a PowerUp
    global addXBlocks
    if elementType == 3:
        worldMap[y][x] = 3
        powerUps.append(PowerUp(x, y, addBlocks=addXBlocks))
    if elementType == 2:
        worldMap[y][x] = 2
        if len(bodyElements) == 0:
            bodyElements.append(BodyElement(x, y, isHead=True))
        else: 
            bodyElements[0] = BodyElement(x, y, isHead=True)
    if elementType == 1:
        worldMap[y][x] = 1
        bodyElements.append(BodyElement(x, y, isHead=False))
    if elementType == 0:
        worldMap[y][x] = 0        
    
def spawnBody() -> None:
    global distToBorder, startBlocks, worldMap   
    
    x = random.randrange(distToBorder, (dimensions - 1) - distToBorder)
    y = random.randrange(distToBorder, (dimensions - 1) - distToBorder)
    
    addBlock(x, y, elementType=2)
    for i in range(startBlocks -1):
        if dir == 1:
            x -= 1
        else:
            x += 1
        addBlock(x, y, elementType=1)
    
def spawnPowerUp(addBlocks: int=1) -> None:
    x = random.randint(0, dimensions -1)
    y = random.randint(0, dimensions -1)
    
    while worldMap[y][x] != 0:
        x = random.randint(0, dimensions -1)
        y = random.randint(0, dimensions -1)
        
    addBlock(x, y, elementType=3)
    powerUps.append(PowerUp(x, y, addBlocks=addBlocks))

def genNewCoord(x:int, y:int, dir:int) -> Tuple[int, int]:
    if dir == 0:
        return x, y -1 
    elif dir == 1:
        return x +1, y
    elif dir == 2:
        return x, y +1
    elif dir == 3:
        return x -1, y
    else:
        raise Exception(f"the provided dir isnt 0, 1, 2, 3 with it being {dir}")

def getHead() -> BodyElement:
    head = bodyElements[0]
    if head.isHead:
        return head
    else:
        errorInLine = line_of_call()
        if errorInLine == -1:
            raise Exception("the head isnt on idx 0 and therefore something is wrong")
        print(f"got error in getHead function with the head at idx not being a head in line {line_of_call()}")
        return BodyElement(-1, -1, isHead=False)

def syncListWithMap():
    global worldMap, bodyElements
    
    worldMap = wipeWorldMap()
    for element in bodyElements:
        if element.isHead:
            worldMap[element.y][element.x] = 2
        else:
            worldMap[element.y][element.x] = 1
            
    for powerUp in powerUps:
        worldMap[powerUp.y][powerUp.x] = 3
            
def moveSnake(dir: int) -> None:
    global worldMap, bodyElements, atePowerUp
    
    head = getHead()
    priorCoord = Coordinates(head.x, head.y)
    newHeadX, newHeadY = genNewCoord(head.x, head.y, dir=dir)
    bodyElements[0].x, bodyElements[0].y = newHeadX, newHeadY
    
    if worldMap[newHeadY][newHeadX] == 3:
        atePowerUp = True
        
    lengthBodyElements = len(bodyElements)
    for idx, element in enumerate(bodyElements):
        if not element.isHead:                
            newX, newY = priorCoord.x, priorCoord.y
            priorCoord.x, priorCoord.y = element.x, element.y
            element.x, element.y = newX, newY
            
            bodyElements[idx] = element
            if atePowerUp and idx == lengthBodyElements -1:
                addBlock(priorCoord.x, priorCoord.y, elementType=1) 
                
    atePowerUp = False                
    syncListWithMap()     
        
def collisionWithBorder(dir:int) -> bool:
    head = getHead()
    currentPos = Coordinates(head.x, head.y)    
    newX, newY = genNewCoord(currentPos.x, currentPos.y, dir=dir)
    if (0 <= newX <= dimensions -1) and (0 <= newY <= dimensions -1): # the new pos is within bounds
        return False
    else:
        return True
    
def collisionWithBody(dir:int) -> bool:
    global worldMap
    head = getHead()
    if head.x == -1 and head.y == -1:
        return True
    
    x, y = genNewCoord(head.x, head.y, dir=dir)
    newCoord = Coordinates(x, y)
    if worldMap[newCoord.y][newCoord.x] in [0, 3]: # it is allowed to go into a void or a powerUp
        return False
    else:
        return True

def modDir() -> int:
    global dir
    return dir

def doGame():
    global lostGame, borderCollsion, bodyCollision, dir, atePowerUp, dimensions, distToBorder, startBlocks, worldMap, addXBlocks, reward
    
    while not lostGame:
        reward += 0.05 # incentive to stay alive
        worldMap = wipeWorldMap()
        spawnBody()
        spawnPowerUp(addBlocks=addXBlocks)
        
        borderCollsion = collisionWithBorder(dir=dir)
        bodyCollision = collisionWithBody(dir)
        if bodyCollision or borderCollsion:
            print(f"lost game with bodyCollision={bodyCollision} and borderCollsion={borderCollsion}")
            lostGame = True
            
        moveSnake(dir=dir)        
        
        dir = modDir()
        