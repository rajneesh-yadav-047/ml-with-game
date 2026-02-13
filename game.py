import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont(None, 35)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100

class SnakeGameAI:
    def __init__(self, w=600, h=400, headless=False):
        self.w = w
        self.h = h
        self.headless = headless
        # init display
        if not self.headless:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Reward for eating food or just moving
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = 0.01 # Small reward for not dying

        if self.head.x == 0 or self.head.x == self.w - BLOCK_SIZE or self.head.y == 0 or self.head.y == self.h - BLOCK_SIZE:
            reward -= 0.5 # Penalize for being at the edge to discourage wall-hugging
        
        # 5. update ui and clock
        if not self.headless:
            self._update_ui()
            self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        if self.headless:
            return
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

class MultiSnakeGameAI:
    def __init__(self, n_snakes=1, w=600, h=600):
        self.w = w
        self.h = h
        self.n_snakes = n_snakes
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Evolution Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snakes = []
        for _ in range(self.n_snakes):
            x = random.randint(2, (self.w - BLOCK_SIZE)//BLOCK_SIZE - 2) * BLOCK_SIZE
            y = random.randint(2, (self.h - BLOCK_SIZE)//BLOCK_SIZE - 2) * BLOCK_SIZE
            head = Point(x, y)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.snakes.append({
                'head': head,
                'body': [head, Point(head.x-BLOCK_SIZE, head.y), Point(head.x-2*BLOCK_SIZE, head.y)],
                'direction': Direction.RIGHT,
                'score': 0,
                'alive': True,
                'color': color,
                'frame_iteration': 0,
                'head_history': []
            })
        
        # Generate common food for all snakes initially
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            common_food = Point(x, y)
            if not any(common_food in s['body'] for s in self.snakes):
                break
        
        for s in self.snakes:
            s['food'] = common_food
            
        self.frame_iteration = 0

    def _place_food(self, snake):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            pt = Point(x, y)
            if pt not in snake['body']:
                snake['food'] = pt
                break

    def play_step(self, actions):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        alive_indices = [i for i, s in enumerate(self.snakes) if s['alive']]
        rewards = []

        # Move all alive snakes
        for i, idx in enumerate(alive_indices):
            snake = self.snakes[idx]
            # Calculate distance before move
            old_dist = abs(snake['head'].x - snake['food'].x) + abs(snake['head'].y - snake['food'].y)
            
            self._move(snake, actions[i])
            snake['body'].insert(0, snake['head'])
            
            # Calculate distance after move
            new_dist = abs(snake['head'].x - snake['food'].x) + abs(snake['head'].y - snake['food'].y)
            
            if new_dist < old_dist:
                snake['step_reward'] = 1
            else:
                snake['step_reward'] = -2

        # Check collisions and food
        for idx in alive_indices:
            snake = self.snakes[idx]
            reward = 0
            snake['frame_iteration'] += 1
            
            # Loop/Spinning Detection
            snake['head_history'].append(snake['head'])
            if len(snake['head_history']) > 60:
                snake['head_history'].pop(0)
            
            # If unique positions in last 60 frames are few, it's likely looping/spinning
            loop_detected = False
            if len(snake['head_history']) > 20 and len(set(snake['head_history'])) < len(snake['head_history']) * 0.6:
                loop_detected = True
            
            # Check collision (Wall or Any Snake)
            if self.is_collision(snake['head']):
                snake['alive'] = False
                reward = -50
            elif loop_detected:
                snake['alive'] = False
                reward = -50
            elif snake['frame_iteration'] > 50*len(snake['body']):
                snake['alive'] = False
                reward = -10
            elif snake['head'] == snake['food']:
                snake['score'] += 1
                reward = 10
                self._place_food(snake)
            else:
                snake['body'].pop()
                reward = 0
            
            reward += snake['step_reward']
            
            rewards.append(reward)

        self._update_ui()
        self.clock.tick(20) # Slower tick to see what's happening
        return rewards

    def is_collision(self, pt):
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        for s in self.snakes:
            if s['alive'] and pt in s['body'][1:]: # Check against all bodies
                 return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for s in self.snakes:
            if s['alive']:
                # Draw specific food for this snake
                pygame.draw.rect(self.display, RED, pygame.Rect(s['food'].x, s['food'].y, BLOCK_SIZE, BLOCK_SIZE))
                for pt in s['body']:
                    pygame.draw.rect(self.display, s['color'], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f"Alive: {sum(1 for s in self.snakes if s['alive'])}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, snake, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake['direction'])

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        snake['direction'] = new_dir
        x = snake['head'].x
        y = snake['head'].y
        if snake['direction'] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif snake['direction'] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif snake['direction'] == Direction.DOWN:
            y += BLOCK_SIZE
        elif snake['direction'] == Direction.UP:
            y -= BLOCK_SIZE
        snake['head'] = Point(x, y)
