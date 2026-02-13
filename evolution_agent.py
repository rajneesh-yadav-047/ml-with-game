import torch
import torch.nn as nn
import numpy as np
import os
import copy
import random
from game import MultiSnakeGameAI, Direction, Point
from plot import plot, plot_weights, plot_network_graph

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU with CUDA support to run.")
    exit()
device = torch.device("cuda")

class EvolutionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        # Disable gradient calculation as we are using Genetic Algorithms
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

    def save(self, file_name='evolution_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

class EvolutionAgent:
    def __init__(self):
        self.input_size = 11
        self.hidden_size = 256
        self.output_size = 3
        self.pop_size = 400
        self.mutation_rate = 0.1
        self.mutation_strength = 0.2
        self.survival_rate = 0.01
        self.population = [EvolutionNet(self.input_size, self.hidden_size, self.output_size).to(device) for _ in range(self.pop_size)]

    @staticmethod
    def get_state(game, snake_idx):
        snake = game.snakes[snake_idx]
        head = snake['head']
        food = snake['food']
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = snake['direction'] == Direction.LEFT
        dir_r = snake['direction'] == Direction.RIGHT
        dir_u = snake['direction'] == Direction.UP
        dir_d = snake['direction'] == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            food.x < head.x,  # food left
            food.x > head.x,  # food right
            food.y < head.y,  # food up
            food.y > head.y  # food down
            ]

        return np.array(state, dtype=int)

    @staticmethod
    def get_action(model, state):
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        prediction = model(state_tensor)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def mutate(self, model):
        child = copy.deepcopy(model)
        for param in child.parameters():
            if len(param.shape) > 1: # Weights
                mask = torch.rand_like(param) < self.mutation_rate
                noise = torch.randn_like(param) * self.mutation_strength
                param.add_(mask * noise)
            else: # Bias
                mask = torch.rand_like(param) < self.mutation_rate
                noise = torch.randn_like(param) * self.mutation_strength
                param.add_(mask * noise)
        return child

    def train_generation(self):
        game = MultiSnakeGameAI(n_snakes=self.pop_size)
        scores = [(0, model, 0) for model in self.population] # (fitness, model, score)
        
        while True:
            alive_indices = [i for i, s in enumerate(game.snakes) if s['alive']]
            if not alive_indices:
                break
                
            # Get states and actions for all alive snakes
            states = [self.get_state(game, i) for i in alive_indices]
            actions = []
            for i, idx in enumerate(alive_indices):
                actions.append(self.get_action(self.population[idx], states[i]))
            
            rewards = game.play_step(actions)
            
            # Update fitness
            for i, idx in enumerate(alive_indices):
                fitness, model, score = scores[idx]
                scores[idx] = (fitness + rewards[i], model, game.snakes[idx]['score'])

        scores.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = scores[0][0]
        best_score = scores[0][2]
        best_model = scores[0][1]
        
        # Selection: Keep top 20% (Elitism)
        survivors_count = int(self.pop_size * self.survival_rate)
        survivors = [s[1] for s in scores[:survivors_count]]
        
        # Reproduction: Fill the rest of the population with mutated children of survivors
        new_population = []
        new_population.extend(survivors) 
        
        while len(new_population) < self.pop_size:
            parent = random.choice(survivors)
            child = self.mutate(parent)
            new_population.append(child)
            
        self.population = new_population
        return best_fitness, best_score, best_model

def train():
    agent = EvolutionAgent()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    generation = 0
    record = 0

    print(f"Starting Evolution Training on {device} ({torch.cuda.get_device_name(0)}) with {agent.pop_size} snakes...")

    while True:
        generation += 1
        best_fitness, best_score, best_model = agent.train_generation()
        
        if best_score > record:
            record = best_score
            best_model.save(f'evolution_model_gen{generation}_s{best_score}.pth')
            print(f"New Record: {record} (Gen {generation})")

        # Update tracking for mean score so the print statement below has a valid value
        plot_scores.append(best_score)
        total_score += best_score
        mean_score = total_score / generation
        plot_mean_scores.append(mean_score)
        # plot(plot_scores, plot_mean_scores)
        # plot_weights(best_model)
        # plot_network_graph(best_model)

        print(f"Gen {generation} | Best Fitness: {best_fitness:.2f} | Best Score: {best_score} | Mean Score: {mean_score:.2f}")

if __name__ == '__main__':
    train()