from env.maze import Maze, UP, RIGHT, DOWN, LEFT
import numpy as np
from pandas import factorize

class MazeEnvironment():

    MAX_STEPS = 1000
    WIN_REWARD = 10
    N_ACTIONS = 4

    __slots__ = ['shape', 'maze', 'current_location', 'goal', 'discrete', 'visit_count', 'steps']

    def __init__(self, shape, discrete=True) -> None:
        # self.max_steps = np.prod(shape) * 10
        self.shape = shape
        self.visit_count = np.zeros(self.shape)
        # self.goal = (self.shape[0] - 1 , self.shape[1] - 1)
        self.discrete = discrete

    def reset(self):
        self.maze = Maze(self.shape)
        self.goal = (
            np.random.randint(self.shape[0] // 3, high=self.shape[0], size=1)[0],
            np.random.randint(self.shape[1] // 3, high=self.shape[1], size=1)[0]
        )

        self.visit_count[:] = 0
        self.visit_count[0, 0] = 1
        self.current_location = (0, 0)
        self.steps = 0
        return self.__observe()

    def step(self, action):
        # action assumed to be legal

        self.steps += 1
        reward = 0
        done = False
        self.current_location = Maze.rel2abs(*self.current_location, action)

        reward -= self.visit_count[self.current_location]
        self.visit_count[self.current_location] += 1

        if self.current_location == self.goal:
            reward += self.WIN_REWARD
            done = True
        elif self.steps > self.MAX_STEPS:
            done = True

        return self.__observe(), reward, done
        

    def legal_actions(self):
        return self.maze.cells[self.current_location]


    def __observe(self):
        neighbours = self.maze.cells[self.current_location]
        state = np.zeros(5)
        for direction in [UP, RIGHT, DOWN, LEFT]:
            if not neighbours[direction]:
                state[direction] = np.inf
            else:
                state[direction] = self.visit_count[Maze.rel2abs(*self.current_location, direction)]
        state[-1] = self.visit_count[self.current_location]
        state = self.__discrete_observe(state) if self.discrete else self.__continuous_observe(state)
        str_state = ""
        for s in state:
            str_state += str(s)
        return str_state
        

    def __continuous_observe(self, state):
        return np.tanh(state)


    def __discrete_observe(self, state):
        return factorize(state, sort=True)[0]
