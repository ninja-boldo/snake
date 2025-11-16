import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Assuming your snake code is in a module called 'snake_engine'
# If testing in same file, this import would be different


class TestWorldMapIndexing(unittest.TestCase):
    """Test correct indexing of worldMap[y][x]"""
    
    def setUp(self):
        """Reset game state before each test"""
        import snake_engine
        snake_engine.dimensions = 10
        snake_engine.worldMap = np.zeros((10, 10))
        snake_engine.bodyElements = []
        snake_engine.powerUps = []
        snake_engine.lostGame = False
        snake_engine.dir = 1
        snake_engine.atePowerUp = False
    
    def test_worldmap_structure(self):
        """Verify worldMap is [y][x] indexing (rows, cols)"""
        from snake_engine import worldMap, dimensions
        self.assertEqual(worldMap.shape, (dimensions, dimensions))
        # worldMap[0] should give us the first row (y=0)
        self.assertEqual(len(worldMap[0]), dimensions)
    
    def test_addblock_correct_indexing(self):
        """Test that addBlock uses worldMap[y][x]"""
        from snake_engine import addBlock, worldMap
        
        # Add a head at position x=3, y=5
        addBlock(x=3, y=5, elementType=2)
        
        # Should be at worldMap[5][3] (row 5, col 3)
        self.assertEqual(worldMap[5][3], 2)
        self.assertEqual(worldMap[3][5], 0)  # Not at [3][5]
    
    def test_sync_list_with_map_indexing(self):
        """Test syncListWithMap uses correct indexing"""
        from snake_engine import bodyElements, BodyElement, syncListWithMap, worldMap
        
        # Manually add body element
        bodyElements.append(BodyElement(x=4, y=6, isHead=True))
        bodyElements.append(BodyElement(x=3, y=6, isHead=False))
        
        syncListWithMap()
        
        # Check head at [y][x] = [6][4]
        self.assertEqual(worldMap[6][4], 2)
        # Check body at [y][x] = [6][3]
        self.assertEqual(worldMap[6][3], 1)
    
    def test_coordinate_consistency(self):
        """Test that x,y coordinates map consistently across functions"""
        from snake_engine import addBlock, worldMap, bodyElements, BodyElement
        
        test_x, test_y = 7, 2
        addBlock(x=test_x, y=test_y, elementType=2)
        
        # Body element should store x=7, y=2
        head = bodyElements[0]
        self.assertEqual(head.x, test_x)
        self.assertEqual(head.y, test_y)
        
        # WorldMap should have value at [y][x] = [2][7]
        self.assertEqual(worldMap[test_y][test_x], 2)


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary and collision detection"""
    
    def setUp(self):
        """Reset state"""
        global dimensions, worldMap, bodyElements, powerUps
        dimensions = 10
        worldMap = np.zeros((dimensions, dimensions))
        bodyElements = []
        powerUps = []
    
    def test_collision_with_border_top(self):
        """Test collision detection at top border"""
        from snake_engine import bodyElements, BodyElement, collisionWithBorder
        
        # Place head at top edge
        bodyElements.append(BodyElement(x=5, y=0, isHead=True))
        
        # Moving up (dir=0) should collide
        self.assertTrue(collisionWithBorder(dir=0))
        
        # Moving right (dir=1) should not collide
        self.assertFalse(collisionWithBorder(dir=1))
    
    def test_collision_with_border_right(self):
        """Test collision at right border"""
        from snake_engine import bodyElements, BodyElement, collisionWithBorder, dimensions
        
        # Place at right edge
        bodyElements.append(BodyElement(x=dimensions-1, y=5, isHead=True))
        
        # Moving right should collide
        self.assertTrue(collisionWithBorder(dir=1))
    
    def test_collision_with_border_bounds_bug(self):
        """Test the off-by-one bug in collisionWithBorder"""
        from snake_engine import bodyElements, BodyElement, collisionWithBorder, dimensions
        
        # The current code checks (0 <= newX < dimensions -1)
        # This means x=dimensions-1 is considered out of bounds!
        # This is a BUG - should be (0 <= newX < dimensions)
        
        bodyElements.append(BodyElement(x=dimensions-2, y=5, isHead=True))
        
        # Moving right to x=dimensions-1 should be VALID but code treats as collision
        result = collisionWithBorder(dir=1)
        # Current buggy behavior: returns True
        # Expected behavior: should return False
        self.assertTrue(result)  # Documents the bug
    
    def test_collision_with_body_self(self):
        """Test collision with snake's own body"""
        from snake_engine import bodyElements, BodyElement, syncListWithMap, collisionWithBody
        
        # Create snake that will collide with itself
        bodyElements.append(BodyElement(x=5, y=5, isHead=True))
        bodyElements.append(BodyElement(x=5, y=6, isHead=False))
        bodyElements.append(BodyElement(x=6, y=6, isHead=False))
        bodyElements.append(BodyElement(x=6, y=5, isHead=False))
        
        syncListWithMap()
        
        # Moving right (dir=1) would hit body at x=6,y=5
        self.assertTrue(collisionWithBody(dir=1))
    
    def test_collision_with_powerup_allowed(self):
        """Test that moving into powerup is allowed"""
        from snake_engine import bodyElements, BodyElement, addBlock, collisionWithBody
        
        bodyElements.append(BodyElement(x=5, y=5, isHead=True))
        addBlock(x=6, y=5, elementType=3)  # PowerUp to the right
        
        # Moving into powerup should NOT be a collision
        self.assertFalse(collisionWithBody(dir=1))


class TestGameLogic(unittest.TestCase):
    """Test core game mechanics"""
    
    def setUp(self):
        global dimensions, worldMap, bodyElements, powerUps, dir, atePowerUp
        dimensions = 10
        worldMap = np.zeros((dimensions, dimensions))
        bodyElements = []
        powerUps = []
        dir = 1
        atePowerUp = False
    
    def test_spawn_body_placement(self):
        """Test that spawning creates body correctly"""
        from snake_engine import spawnBody, bodyElements, worldMap, startBlocks
        
        with patch('snake_engine.random.randrange') as mock_rand:
            mock_rand.return_value = 5
            spawnBody()
        
        # Should have startBlocks elements
        self.assertEqual(len(bodyElements), startBlocks)
        
        # First element should be head
        self.assertTrue(bodyElements[0].isHead)
        
        # Check worldMap has head (value 2) and body (value 1)
        head = bodyElements[0]
        self.assertEqual(worldMap[head.y][head.x], 2)
    
    def test_spawn_powerup_bounds_bug(self):
        """Test the hardcoded bounds bug in spawnPowerUp"""
        from snake_engine import spawnPowerUp, dimensions
        
        # Current code uses random.randint(0, 19) regardless of dimensions
        # With dimensions=10, this will cause IndexError!
        
        with self.assertRaises(IndexError):
            with patch('snake_engine.random.randint') as mock_rand:
                mock_rand.return_value = 15  # Out of bounds for 10x10
                spawnPowerUp()
    
    def test_gen_new_coord_directions(self):
        """Test coordinate generation for all directions"""
        from snake_engine import genNewCoord
        
        x, y = 5, 5
        
        # Up (0): y-1
        new_x, new_y = genNewCoord(x, y, 0)
        self.assertEqual((new_x, new_y), (5, 4))
        
        # Right (1): x+1
        new_x, new_y = genNewCoord(x, y, 1)
        self.assertEqual((new_x, new_y), (6, 5))
        
        # Down (2): y+1
        new_x, new_y = genNewCoord(x, y, 2)
        self.assertEqual((new_x, new_y), (5, 6))
        
        # Left (3): x-1
        new_x, new_y = genNewCoord(x, y, 3)
        self.assertEqual((new_x, new_y), (4, 5))
    
    def test_move_snake_basic(self):
        """Test basic snake movement"""
        from snake_engine import bodyElements, BodyElement, moveSnake, syncListWithMap
        
        # Create simple 2-element snake
        bodyElements.append(BodyElement(x=5, y=5, isHead=True))
        bodyElements.append(BodyElement(x=4, y=5, isHead=False))
        syncListWithMap()
        
        # Move right (dir=1)
        moveSnake(dir=1)
        
        # Head should now be at x=6, y=5
        head = bodyElements[0]
        self.assertEqual(head.x, 6)
        self.assertEqual(head.y, 5)
        
        # Body should be at old head position
        body = bodyElements[1]
        self.assertEqual(body.x, 5)
        self.assertEqual(body.y, 5)
    
    def test_move_snake_eats_powerup(self):
        """Test snake grows when eating powerup"""
        from snake_engine import bodyElements, BodyElement, addBlock, moveSnake, atePowerUp
        
        # Snake at x=5, body at x=4
        bodyElements.append(BodyElement(x=5, y=5, isHead=True))
        bodyElements.append(BodyElement(x=4, y=5, isHead=False))
        
        # PowerUp at x=6 (where head will move)
        addBlock(x=6, y=5, elementType=3)
        
        initial_length = len(bodyElements)
        moveSnake(dir=1)
        
        # Snake should have grown
        self.assertGreater(len(bodyElements), initial_length)


class TestWorldMapStates(unittest.TestCase):
    """Test worldMap state management"""
    
    def test_wipe_world_map_inplace(self):
        """Test wiping worldMap in place"""
        from snake_engine import worldMap, wipeWorldMap
        
        worldMap[5][5] = 2
        wipeWorldMap(inplace=True)
        
        self.assertEqual(worldMap[5][5], 0)
        self.assertTrue(np.all(worldMap == 0))
    
    def test_wipe_world_map_return(self):
        """Test wiping worldMap returns new array"""
        from snake_engine import worldMap, wipeWorldMap
        
        worldMap[5][5] = 2
        new_map = wipeWorldMap(inplace=False)
        
        # Original should be unchanged
        self.assertEqual(worldMap[5][5], 2)
        # New should be zeros
        self.assertTrue(np.all(new_map == 0))
    
    def test_world_map_values(self):
        """Test worldMap uses correct values for different elements"""
        from snake_engine import addBlock, worldMap
        
        # 0 = void/empty
        addBlock(x=1, y=1, elementType=0)
        self.assertEqual(worldMap[1][1], 0)
        
        # 1 = body
        addBlock(x=2, y=2, elementType=1)
        self.assertEqual(worldMap[2][2], 1)
        
        # 2 = head
        addBlock(x=3, y=3, elementType=2)
        self.assertEqual(worldMap[3][3], 2)
        
        # 3 = powerup
        addBlock(x=4, y=4, elementType=3)
        self.assertEqual(worldMap[4][4], 3)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)