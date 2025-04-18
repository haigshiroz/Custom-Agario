# CS5100 Foundation of AI Final Project: AI Agent for Agar.io

## Project Overview
We implemented a Q-learning-based AI agent to autonomously play Agar.io. Our agent was designed to learn strategies for mass accumulation through reward-based learning.

## Experimental Branch: NewModel
We explored an alternative initialization approach in a separate branch:

**Key Differences from Main Branch**:
- Guided Q-table initialization with:
  ```python
  # Priority-based initial values
  FOOD_REWARD = 30       # Slightly > cell reward (20)
  ESCAPE_REWARD = 50     # Significantly > idle penalty (-1) 
  HUNT_REWARD = 1200     # Slightly > player reward (1000)
  
## How to Reproduce
```bash
python multiple_instance.py 

```
