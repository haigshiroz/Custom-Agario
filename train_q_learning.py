import pandas as pd
import numpy as np
import pickle
import random

# --- CONFIGURATION ---
EXCEL_PATH = "state_scores.xlsx"      # The excel file with rough data
Q_TABLE_PATH = "Q_table.pickle"       # Output file for the Q-table
NUM_ACTIONS = 6                       # kept it at 6 due to 6 action states

ALPHA = 0.1                           # Learning rate factor taken at rought as of now
GAMMA = 0.95                          # Discount factor(it is also a rough estiamte)
EPISODES = 100                        # Total episodes to train
MAX_STEPS_PER_EPISODE = 10           # Steps per episode

# Loading our data from excel
df = pd.read_excel(EXCEL_PATH)
states = df['State'].tolist()
scores = df['Score'].tolist()

max_state = max(states)
q_table = np.zeros((max_state + 1, NUM_ACTIONS))

# Look up for rewards
reward_lookup = dict(zip(states, scores))

# Training data
for episode in range(EPISODES):
    state = random.choice(states)

    for step in range(MAX_STEPS_PER_EPISODE):
        epsilon = 0.1
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = np.argmax(q_table[state])

        reward = reward_lookup.get(state, 0)

        if random.random() < 0.5:
            next_state = min(max_state, state + random.choice([-1, 1]))
        else:
            next_state = state

        # Updating the given Q-Learning
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state][action] = new_value

        state = next_state

print(" Episodic Q-learning training complete.")

# Saving trained table
with open(Q_TABLE_PATH, "wb") as f:
    pickle.dump(q_table, f)

print(f" Trained Q-table saved to: {"C:\Users\prana\OneDrive\Desktop\FAI_New\Custom-Agario\q_table.py"}" )