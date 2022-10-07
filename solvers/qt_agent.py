import numpy as np
from env.maze_env import MazeEnvironment


class QTAgent():

    __slots__ = ['lr', 'discount', 'q_table']

    def __init__(self, lr=0.1, discount=0.99, q_table=None) -> None:
        self.q_table = dict() if q_table is None else q_table
        self.predict("00001")
        self.lr = lr
        self.discount = discount



    def update(self, transition):
        current_state, action, reward, next_state, done = transition
        max_future_q = np.max(self.predict(next_state))
        current_q = self.predict(current_state)[action]
        if not done:
            new_q = (1 - self.lr) * current_q + self.lr * (reward + self.discount * max_future_q)
        else:
            new_q = reward
        self.q_table[current_state][action] = new_q


    def predict(self, state):
        if state not in self.q_table:
            self.q_table[state] = list(np.random.uniform(low=0, high=0, size=MazeEnvironment.N_ACTIONS))
        return self.q_table[state]