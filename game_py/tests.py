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
        import snake_engine
        
        # Manually add body element
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=4, y=6, isHead=True))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=3, y=6, isHead=False))
        
        snake_engine.syncListWithMap()
        
        # Check head at [y][x] = [6][4]
        self.assertEqual(snake_engine.worldMap[6][4], 2)
        # Check body at [y][x] = [6][3]
        self.assertEqual(snake_engine.worldMap[6][3], 1)
    
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
        import snake_engine
        snake_engine.dimensions = 10
        snake_engine.worldMap = np.zeros((10, 10))
        snake_engine.bodyElements = []
        snake_engine.powerUps = []
    
    def test_collision_with_border_top(self):
        """Test collision detection at top border"""
        import snake_engine
        
        # Clear any existing elements
        snake_engine.bodyElements = []
        
        # Place head at top edge
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=0, isHead=True))
        
        # Moving up (dir=0) should collide (y-1 = -1, out of bounds)
        self.assertTrue(snake_engine.collisionWithBorder(dir=0))
        
        # Moving right (dir=1) should not collide
        self.assertFalse(snake_engine.collisionWithBorder(dir=1))
    
    def test_collision_with_border_right(self):
        """Test collision at right border"""
        import snake_engine
        
        # Clear any existing elements
        snake_engine.bodyElements = []
        
        # Place at right edge
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=snake_engine.dimensions-1, y=5, isHead=True))
        
        # Moving right should collide (x+1 = 10, out of bounds for 0-9)
        self.assertTrue(snake_engine.collisionWithBorder(dir=1))
    
    def test_collision_with_border_bounds_bug(self):
        """Test boundary checking works correctly"""
        import snake_engine
        
        # Place at second-to-last column
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=snake_engine.dimensions-2, y=5, isHead=True))
        
        # Moving right to x=dimensions-1 should be VALID (not a collision)
        result = snake_engine.collisionWithBorder(dir=1)
        self.assertFalse(result)  # Should be False since dimensions-1 is valid
    
    def test_collision_with_body_self(self):
        """Test collision with snake's own body"""
        import snake_engine
        
        # Create snake that will collide with itself
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=5, isHead=True))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=6, isHead=False))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=6, y=6, isHead=False))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=6, y=5, isHead=False))
        
        snake_engine.syncListWithMap()
        
        # Moving right (dir=1) would hit body at x=6,y=5
        self.assertTrue(snake_engine.collisionWithBody(dir=1))
    
    def test_collision_with_powerup_allowed(self):
        """Test that moving into powerup is allowed"""
        import snake_engine
        
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=5, isHead=True))
        snake_engine.addBlock(x=6, y=5, elementType=3)  # PowerUp to the right
        
        # Moving into powerup should NOT be a collision
        self.assertFalse(snake_engine.collisionWithBody(dir=1))


class TestGameLogic(unittest.TestCase):
    """Test core game mechanics"""
    
    def setUp(self):
        import snake_engine
        snake_engine.dimensions = 10
        snake_engine.worldMap = np.zeros((10, 10))
        snake_engine.bodyElements = []
        snake_engine.powerUps = []
        snake_engine.dir = 1
        snake_engine.atePowerUp = False
        snake_engine.startBlocks = 3
    
    def test_spawn_body_placement(self):
        """Test that spawning creates body correctly"""
        import snake_engine
        
        # Clear any existing body elements first
        snake_engine.bodyElements = []
        
        with patch('random.randrange') as mock_rand:
            mock_rand.return_value = 5
            snake_engine.spawnBody()
        
        # Should have startBlocks elements
        self.assertEqual(len(snake_engine.bodyElements), snake_engine.startBlocks)
        
        # First element should be head
        self.assertTrue(snake_engine.bodyElements[0].isHead)
        
        # Check worldMap has head (value 2) and body (value 1)
        head = snake_engine.bodyElements[0]
        self.assertEqual(snake_engine.worldMap[head.y][head.x], 2)
    
    def test_spawn_powerup_bounds_bug(self):
        """Test that spawnPowerUp uses correct dimensions"""
        import snake_engine
        
        # Should not raise IndexError with dimensions=10
        # Multiple attempts to ensure we don't get lucky with random
        for _ in range(10):
            snake_engine.powerUps = []  # Clear between attempts
            snake_engine.wipeWorldMapInplace()
            try:
                snake_engine.spawnPowerUp()
                # Verify powerup is within bounds
                if snake_engine.powerUps:
                    pu = snake_engine.powerUps[-1]
                    self.assertLess(pu.x, snake_engine.dimensions)
                    self.assertLess(pu.y, snake_engine.dimensions)
            except IndexError:
                self.fail("spawnPowerUp raised IndexError with dimensions=10")
    
    def test_gen_new_coord_directions(self):
        """Test coordinate generation for all directions"""
        import snake_engine
        
        x, y = 5, 5
        
        # Up (0): y-1
        new_x, new_y = snake_engine.genNewCoord(x, y, 0)
        self.assertEqual((new_x, new_y), (5, 4))
        
        # Right (1): x+1
        new_x, new_y = snake_engine.genNewCoord(x, y, 1)
        self.assertEqual((new_x, new_y), (6, 5))
        
        # Down (2): y+1
        new_x, new_y = snake_engine.genNewCoord(x, y, 2)
        self.assertEqual((new_x, new_y), (5, 6))
        
        # Left (3): x-1
        new_x, new_y = snake_engine.genNewCoord(x, y, 3)
        self.assertEqual((new_x, new_y), (4, 5))
    
    def test_move_snake_basic(self):
        """Test basic snake movement"""
        import snake_engine
        
        # Create simple 2-element snake
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=5, isHead=True))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=4, y=5, isHead=False))
        snake_engine.syncListWithMap()
        
        # Move right (dir=1)
        snake_engine.moveSnake(dir=1)
        
        # Head should now be at x=6, y=5 (moved right)
        head = snake_engine.bodyElements[0]
        self.assertEqual(head.x, 6)  # ✅ Correct behavior - head moves!
        self.assertEqual(head.y, 5)
        
        # Body should be at old head position
        body = snake_engine.bodyElements[1]
        self.assertEqual(body.x, 5)
        self.assertEqual(body.y, 5)
    
    def test_move_snake_eats_powerup(self):
        """Test snake grows when eating powerup"""
        import snake_engine
        
        # Clear existing state
        snake_engine.bodyElements = []
        snake_engine.powerUps = []
        snake_engine.wipeWorldMapInplace()
        
        # Snake at x=5, body at x=4
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=5, y=5, isHead=True))
        snake_engine.bodyElements.append(snake_engine.BodyElement(x=4, y=5, isHead=False))
        snake_engine.syncListWithMap()
        
        # PowerUp at x=6 (where head will move)
        snake_engine.addBlock(x=6, y=5, elementType=3)
        
        initial_length = len(snake_engine.bodyElements)
        snake_engine.moveSnake(dir=1)
        
        # Snake should have grown by 1
        self.assertEqual(len(snake_engine.bodyElements), initial_length + 1)


class TestWorldMapStates(unittest.TestCase):
    """Test worldMap state management"""
    
    def test_wipe_world_map_inplace(self):
        """Test wiping worldMap in place"""
        import snake_engine
        
        snake_engine.worldMap[5][5] = 2
        snake_engine.wipeWorldMapInplace()
        
        self.assertEqual(snake_engine.worldMap[5][5], 0)
        self.assertTrue(np.all(snake_engine.worldMap == 0))
    
    def test_wipe_world_map_return(self):
        """Test wiping worldMap returns new array"""
        import snake_engine
        
        snake_engine.worldMap[5][5] = 2
        new_map = snake_engine.wipeWorldMap()
        
        # Original should be unchanged
        self.assertEqual(snake_engine.worldMap[5][5], 2)
        # New should be zeros
        self.assertTrue(np.all(new_map == 0))
    
    def test_world_map_values(self):
        """Test worldMap uses correct values for different elements"""
        import snake_engine
        
        # 0 = void/empty
        snake_engine.addBlock(x=1, y=1, elementType=0)
        self.assertEqual(snake_engine.worldMap[1][1], 0)
        
        # 1 = body (requires head to exist first)
        snake_engine.bodyElements = []  # Clear first
        snake_engine.addBlock(x=3, y=3, elementType=2)  # Add head first
        snake_engine.addBlock(x=2, y=2, elementType=1)  # Then body
        self.assertEqual(snake_engine.worldMap[2][2], 1)
        
        # 2 = head
        self.assertEqual(snake_engine.worldMap[3][3], 2)
        
        # 3 = powerup
        snake_engine.addBlock(x=4, y=4, elementType=3)
        self.assertEqual(snake_engine.worldMap[4][4], 3)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)