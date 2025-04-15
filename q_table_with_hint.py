import pickle
import random

# Guided values aligned with your reward structure
GUIDE_VALUES = {
    'food': 30,      # Slightly > cell reward (20)
    'escape': 50,    # Significantly > idle penalty (-1)
    'hunt': 1200     # Slightly > player reward (1000)
}

def custom_hash(state):
    """Hash function with most_food strictly 0-3 (4 directions)"""
    assert 0 <= state['most_food'] <= 3, "most_food must be 0-3"
    return (state['most_food'] * 25) + (state['largest_bigger'] * 5) + state['largest_smaller']

def decode_state(state_hash):
    """Extracts original state features from hash"""
    return {
        'food_dir': (state_hash // 25) % 4,    # 0-3 only
        'threat_dir': (state_hash // 5) % 5,   # 0-4
        'prey_dir': state_hash % 5             # 0-4
    }

def get_guided_action(state_hash):
    """Determines suggested action based on game priorities"""
    state = decode_state(state_hash)
    
    # Priority 1: Escape threats
    if state['threat_dir'] != 4:
        return (state['threat_dir'] + 2) % 4  # Opposite direction
    
    # Priority 2: Hunt prey
    if state['prey_dir'] != 4:
        return state['prey_dir']
    
    # Priority 3: Seek food (food_dir is always 0-3)
    return state['food_dir']  # No fallback needed since food_dir always exists

# Initialize Q-tables
NUM_STATES = 100  # 4 (food) × 5 (threat) × 5 (prey)
NUM_ACTIONS = 4    # Up, Right, Down, Left

Q = {}
Q_num_updates = {}

for state_hash in range(NUM_STATES):
    guided_action = get_guided_action(state_hash)
    state = decode_state(state_hash)
    
    Q[state_hash] = []
    Q_num_updates[state_hash] = []
    
    for action in range(NUM_ACTIONS):
        # Base random value for exploration
        base_value = random.uniform(-0.1, 0.1)
        
        # Boost guided actions
        if action == guided_action:
            if state['threat_dir'] != 4:
                Q[state_hash].append(base_value + GUIDE_VALUES['escape'])
            elif state['prey_dir'] != 4:
                Q[state_hash].append(base_value + GUIDE_VALUES['hunt'])
            else:
                Q[state_hash].append(base_value + GUIDE_VALUES['food'])
        else:
            Q[state_hash].append(base_value)
        
        Q_num_updates[state_hash].append(0)

# Save initialization
with open('Q_table.pickle', 'wb') as f:
    pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('Q_num_updates.pickle', 'wb') as handle:
    pickle.dump(Q_num_updates, handle, protocol=pickle.HIGHEST_PROTOCOL)