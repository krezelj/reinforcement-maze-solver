from tqdm import tqdm
from env.maze_env import MazeEnvironment
from solvers.qt_agent import QTAgent
import numpy as np
import json

epsilon = 1.0
EPSILON_DECAY = 0.9999
LEGAL_DELTA = 0.001

MAX_EPISODES = 1
SHOW_EVERY = 1

LOAD_Q_TABLE = True


env = MazeEnvironment((15, 15))

if LOAD_Q_TABLE:
    with open('models/qt_8x8_random_goal.json', 'r') as file:
        q_table = json.load(file)
    agent = QTAgent(q_table=q_table)
else:
    agent = QTAgent()




for episode in tqdm(range(1, MAX_EPISODES+1), ascii=True, unit='episodes'):

    current_state = env.reset()
    done = False

    while not done:
        if np.random.random() > epsilon:
            qs = agent.predict(current_state)
            qs -= np.min(qs) - LEGAL_DELTA
            qs *= env.legal_actions()
            action = np.argmax(qs)
        else:
            action = np.argmax(np.random.random(size=env.N_ACTIONS) * env.legal_actions())
        
        next_state, reward, done = env.step(action)
        transition = current_state, action, reward, next_state, done
        agent.update(transition)

        current_state = next_state
        epsilon *= EPSILON_DECAY

    if episode % SHOW_EVERY == 0:
        for _ in range(10):
            current_state = env.reset()
            done = False
            path = []
            while not done:
                qs = agent.predict(current_state)
                qs -= np.min(qs) - LEGAL_DELTA
                qs *= env.legal_actions()
                action = np.argmax(qs)

                path.append(env.current_location)
                next_state, _, done = env.step(action)
                current_state = next_state
            path.append(env.current_location)
            for cell in path[-10:]:
                print(cell)
            env.maze.render(path=path, animate=True, goal=env.goal)

with open('models/qt_8x8_random_goal.json', 'w') as file:
     file.write(json.dumps(agent.q_table))