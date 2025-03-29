import pickle
import random

# Initialize tables
Q_table = {}
Q_num_updates = {}

# Fill them with dummy values
# First 100 states are during the game, state 101 is player died
for temp_state in range(101):
    # Initialize action dictionaries
    Q_table[temp_state] = []
    Q_num_updates[temp_state] = []
    # For each action
    for a in range(5):
        # Initialize values to 0
        Q_table[temp_state].append(0.0)
        # Q_table[temp_state].append(random.random()) # Just to test
        Q_num_updates[temp_state].append(0)

with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('Q_num_updates.pickle', 'wb') as handle:
    pickle.dump(Q_num_updates, handle, protocol=pickle.HIGHEST_PROTOCOL)

