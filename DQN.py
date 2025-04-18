# Imports & Constants
import random
import pyautogui
import time
import math
import json 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms

from PIL import Image


# --- Pyautogui Constants ---
pyautogui.FAILSAFE = True

# Screen dimensions: (2879, 1799)
screen_width, screen_height = pyautogui.size()

PLAY_BUTTON_COORD = (screen_width * 1/2, 731)
up_coord = (screen_width * 1/2, screen_height * 1/4)
right_coord = (screen_width * 3/4, screen_height * 1/2)
down_coord = (screen_width * 1/2, screen_height * 3/4)
left_coord = (screen_width * 1/4, screen_height * 1/2)

up_right_coord = (screen_width * 3/4, screen_height * 1/4)
down_right_coord = (screen_width * 3/4, screen_height * 3/4)
down_left_coord = (screen_width * 1/4, screen_height * 3/4)
up_left_coord = (screen_width * 1/4, screen_height * 1/4)

center = (screen_width * 1/2, screen_height * 1/2)

DIRECTIONS = [up_coord, right_coord, down_coord, left_coord, up_right_coord, down_right_coord, down_left_coord, up_left_coord, center]

PLAY_BUTTON_COORD = (screen_width * 1/2, 731)


# --- DQN Constants ---
GAMMA = 0.95 # Discount factor
LEARNING_RATE = 0.0001 # For gradient descent
EPSILON = 1  # Randomness probability
DECAY_RATE = 0.9998 # Decay rate for randomness, use 0.9975 for 100 lives, use 0.999 for 200 lives, 9993 for 300 lives, 0.9998 for 600 lives
NUM_ACTIONS = len(DIRECTIONS) + 1 # + 1 is for splitting

FRAMES_PER = 2 # How many frames are inputted to the model to convey motion

PREPROCESS_WIDTH = 120
PREPROCESS_HEIGHT = 53






'''
DQN Model

Architecture:
Layer 1:
- Convolution: 4 frames -> 32, kernel size of 3
- ReLU
- Pool - kernel size of 2, stride of 2

Layer 2:
- Convolution: 32 -> 64, kernel size of 3
- ReLU
- Pool - kernel size of 2, stride of 2

Flattened:
- Linear: 64 * new width * new height -> 5 (number actions)
'''
class DQN(nn.Module):
    '''
    n_frames: Number of frames (in order to convey motion)
    n_actions: Number of output actions you can take
    '''
    def __init__(self, n_frames, n_actions, width, height):
        super(DQN, self).__init__()

        pool_stride = 2

        # --- First layer ---
        # in_channels/n_frames is equivalent to RGB. Here, we want to look at multiple frames for motion
        # out_channels is how many kernels we apply
        # kernel_size is the size of each kernel (ex: the smaller window shadowing the larger window)
        # stride is how many pixels the kernel moves by
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=8, kernel_size=6, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=pool_stride)

        # --- Second layer ---
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=6, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=pool_stride)

        # --- Flattened layer ---
        fc_flattened = 576 # Output of second layer

        self.fc1 = nn.Linear(fc_flattened, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        # --- First layer ---
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # --- Second layer ---
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # --- Flattened layer ---
        # Reshape output
        x = x.view(x.size()[0], -1)
        # Pass through the fully connected layer
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # --- Return ---
        return x






# --- Interacting 

'''
Executes given action
0 = up, 1 = right, 2 = down, 3 = left, 4 = shoot
'''
def move_agario(action: int):
    if (action != NUM_ACTIONS - 1):
        pyautogui.moveTo(DIRECTIONS[action]) 
    else:
        pyautogui.press('space')


# Preprocess
def preprocess(img):
    mean_and_std = [0.5] * FRAMES_PER

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((PREPROCESS_HEIGHT, PREPROCESS_WIDTH)),
        transforms.ToTensor(),
        # Normalizes numbers to be to -1 to 1 instead of 0 to 1
        transforms.Normalize(mean=mean_and_std, std=mean_and_std)  
    ])
    # Apply the transformations
    transformed_image = transform(img)
    return transformed_image

def agario_died() -> bool:
    dead = True

    # Take colored screenshot
    agario_sc = pyautogui.screenshot(region=(0, 243, screen_width, 1277))
    img_np = np.array(agario_sc)

    # Iterate through each pixel
    height, width = img_np.shape[:2]
    for y in range(0, height, 5):
        for x in range(0, width, 5):
            # If the pixel is the non-tinted background, we are not dead
            pixel = img_np[y, x]
            if (list(pixel) == [242, 251, 255]):
                dead = False
                break
    return dead


# Take screenshot of game
def capture_and_preprocess_agario_sc():
    # Take FRAMES_PER screenshots of game to convey motion
    # Convert to black and white
    frames = []
    for _ in range(FRAMES_PER):
        agario_sc = pyautogui.screenshot(region=(0, 243, screen_width, 1277)).convert('L')
        frames.append(agario_sc)
        time.sleep(0.1)

    # Stack them together
    # Weird hack but use RGBA to do so (or LA for only 2 channels)
    image_with_four_frames = Image.merge("LA", tuple(frames))

    # Preprocess to be same dimensions
    preprocessed_image = preprocess(image_with_four_frames)

    return preprocessed_image






# --- Training using DQN Techniques ---

previous_action = None
previous_action_count = 0

'''
Trains the model using DQN discussed in class

Q-hat is Epsilon-greedy output of the predict neural network

Target Q value is reward + (Gamma * V_opt) where V_opt is the optimal output of target NN

Get action from Q-hat, get reward and new state, then compute MSE loss

Returns whether the episode is done or not
'''
def train(predict_nn: DQN, target_nn: DQN, optimizer: torch.optim, mse: list) -> bool:
    # Declare global variables
    global EPSILON
    global DECAY_RATE
    global previous_action
    global previous_action_count
    
    # Zero the gradients
    optimizer.zero_grad()

    # Get the current state and turn into a batch which we turn into a state
    state_img = capture_and_preprocess_agario_sc() 
    batch = torch.stack([state_img]) # TODO Can increase batch size here
    state = torch.tensor(batch, dtype=torch.float32)

    # Forward pass with State into first nn
    # Gets Q-values for all actions
    q_values_predicted = predict_nn(state) 

    # Epsilon-greedy policy to get random or optimal
    if random.random() < EPSILON:
        # Actual action - value in range [0, 4] inclusive
        action = random.randint(0, NUM_ACTIONS - 1) 
        # Q value for random action. We take [0] to get the value instead of [value]
        predicted_q_value = q_values_predicted[:, action][0] 
    else:
        # Q value for the best action
        predicted_q_value = q_values_predicted.max() 
        # Actual action - value in range [0, 4] inclusive
        action = q_values_predicted.argmax().item() 

    if (previous_action == action):
        previous_action_count += 1
    else:
        previous_action = action
        previous_action_count = 0


    # Execute action
    move_agario(action)

    # Get reward R
    reward = 1
    died = agario_died()
    if (died):
        reward = -100
    if (previous_action_count >= 3):
        reward -= 5

    # Get new state S'
    new_state_img = capture_and_preprocess_agario_sc() 
    new_batch = torch.stack([new_state_img]) # Can increase batch size here
    new_state = torch.tensor(new_batch, dtype=torch.float32)


    # Forward pass with S' to get targe Q value
    # Don't update weights for the target NN, we will copy over weights later
    with torch.no_grad():
        # Gets Q-values for all actions
        q_values_target = target_nn(new_state)
        # Get best action for V_opt
        target_q_value = q_values_target.max()

    # Get the future reward q value
    target_q_value = reward + (GAMMA * target_q_value)

    # Loss + optimize
    loss = F.mse_loss(predicted_q_value, target_q_value)
    mse.append(loss.item())
    loss.backward()
    optimizer.step()

    # Decay epsilon
    EPSILON *= DECAY_RATE
    
    return died






'''
Graphs the MSE and line of best fit
'''
def graph_mse(mse):
    
    x_mse = np.arange(len(mse))
    m_mse, b_mse = np.polyfit(x_mse, mse, 1)
    plt.plot(x_mse, m_mse*x_mse + b_mse, color='black', label='Mean Squared Error Loss')

    plt.plot(mse)
    plt.title("Mean Squared Error vs Actions Taken")
    plt.xlabel("Action Number")
    plt.ylabel("Mean Squared Error Loss of Q-Values")
    plt.show()


'''
Graphs the time alive and line of best fit
'''
def graph_time_alive(time_alive):
    x_time_alive = np.arange(len(time_alive))
    m_time_alive, b_time_alive = np.polyfit(x_time_alive, time_alive, 1)
    plt.plot(time_alive)
    plt.plot(x_time_alive, m_time_alive*x_time_alive + b_time_alive, color='black', label='Time Alive')
    plt.title("Time Alive (s) vs Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Time Alive (s)")






mse = []
time_alive = []

'''
Core training loop.
Instantiates predict neural network, target neural network, and optimizer
Then swaps tabs to Agar.io open and starts training for the given number of episodes
Lastly graphs mean squared error and time alive
'''
def run_training():
    global mse
    global time_alive
    global EPSILON
    global DECAY_RATE

    # Instantiate both models
    predict_nn = DQN(n_frames=FRAMES_PER, n_actions=NUM_ACTIONS, width=PREPROCESS_WIDTH, height=PREPROCESS_HEIGHT) # Theta 1
    target_nn = DQN(n_frames=FRAMES_PER, n_actions=NUM_ACTIONS, width=PREPROCESS_WIDTH, height=PREPROCESS_HEIGHT) # Theta 2    
    predict_nn.load_state_dict(torch.load('./SAVED/predict_nn_weights-433.pth'))
    
    # Have them start with the same weights
    target_nn.load_state_dict(predict_nn.state_dict())

    # Intantiate other training objects
    optimizer = optim.Adam(predict_nn.parameters(), lr=LEARNING_RATE)

    # Tab to agario
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)
    pyautogui.keyDown('enter')
    time.sleep(0.5)


    for i in range(433):
        EPSILON *= DECAY_RATE

    try:
        # 1 episode = full life (spawn in -> death)
        num_episodes = 75
        for i in range(num_episodes):
            print(f'Life {i}')
            time_start = time.time()
            died = False

            # Update Target weights to Predicted Weights every 5 episodes
            if (num_episodes % 10 == 0):
                target_nn.load_state_dict(predict_nn.state_dict())

            # Go to play button & press play
            pyautogui.moveTo(PLAY_BUTTON_COORD)
            pyautogui.click()
            time.sleep(0.75)

            while(not died):
                died = train(predict_nn, target_nn, optimizer, mse)
    
            # Reset the game if died
            if (died):
                # While the start button is not present, loop
                start_game_pixel = None
                while (start_game_pixel != (52, 127, 1) and start_game_pixel != (84, 200, 0)):
                    time.sleep(0.1) 
                    # Take another screenshot and check again
                    start_game_sc = pyautogui.screenshot(region=(1344, 731, 1,1))
                    start_game_pixel = start_game_sc.getpixel((0, 0))

                time_alive.append(time.time() - time_start)

    except Exception as e:
        print(e)
        

    # After training, graph MSE & time alive
    graph_mse(mse)
    graph_time_alive(time_alive)


    # Save the weights
    torch.save(predict_nn.state_dict(), './predict_nn_weights.pth')


run_training()





# Saves statistics
with open('time_alive.json', 'w') as f:
    json.dump(time_alive, f)

with open('mse.json', 'w') as f:
    json.dump(mse, f)


# Loads saved statistics
with open('./SAVED/mse-incomplete.json', 'r') as f:
    mse = json.load(f)

with open('./SAVED/time_alive-incomplete.json', 'r') as f:
    time_alive = json.load(f)






# --- Training using DQN Techniques ---

'''
Helper for running the main run-without-training loop to take a screenshot, run through the DQN, 
get an action, play it, and return whether the agent died (end of episode) or not
'''
def screenshot_and_play(agario_bot: DQN) -> bool:
    # Get the current state and turn into a batch which we turn into a state
    state_img = capture_and_preprocess_agario_sc() 
    batch = torch.stack([state_img])
    state = torch.tensor(batch, dtype=torch.float32)

    # Get the q_values without updating the gradients
    with torch.no_grad():
        q_values_predicted = agario_bot(state) 

    # Actual action - value in range [0, 4] inclusive
    action = q_values_predicted.argmax().item() 

    # Execute action
    move_agario(action)

    # Return whether end of episode
    died = agario_died()
    return died



'''
Runs the model without training the weights
Loads a saved model, alt tabs into the game, and runs foreverx
'''
def run_model_without_training():

    # Instantiate both models
    agario_bot = DQN(n_frames=FRAMES_PER, n_actions=NUM_ACTIONS, width=PREPROCESS_WIDTH, height=PREPROCESS_HEIGHT) # Theta 1
    agario_bot.load_state_dict(torch.load('./SAVED/predict_nn_weights.pth'))
    
    # Tab to agario
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)
    pyautogui.keyDown('enter')
    time.sleep(0.5)


    try:
        while True:
            time_start = time.time()
            died = False

            # Go to play button & press play
            pyautogui.moveTo(PLAY_BUTTON_COORD)
            pyautogui.click()
            time.sleep(0.75)

            # Play while not dead
            while(not died):
                died = screenshot_and_play(agario_bot)
    
            # Reset the game if died

            # While the start button is not present, loop
            start_game_pixel = None
            while (start_game_pixel != (52, 127, 1) and start_game_pixel != (84, 200, 0)):
                time.sleep(0.1)
                # Take another screenshot and check again
                start_game_sc = pyautogui.screenshot(region=(1344, 731, 1,1))
                start_game_pixel = start_game_sc.getpixel((0, 0))

            print(f"Time alive: {time.time() - time_start}")

    except Exception as e:
        print(e)


run_model_without_training()





