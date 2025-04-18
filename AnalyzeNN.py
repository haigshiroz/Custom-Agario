# Imports & Constants
import pyautogui
import time

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

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






class DQN(nn.Module):
    '''
    n_frames: Number of frames (in order to convey motion)
    n_actions: Number of output actions you can take
    '''
    def __init__(self, n_frames):
        super(DQN, self).__init__()

        pool_stride = 2

        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=8, kernel_size=6, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=pool_stride)

        # --- Second layer ---
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=6, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=pool_stride)

    def forward(self, x):
        # --- First layer ---
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # --- Second layer ---
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # --- Return ---
        return x



# Preprocess
def new_preprocess(img):
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


def show_preprocessed_image(tensor_image):
    mean_and_std = [0.5] * tensor_image.shape[0]  # Same as used during preprocessing

    # Reverse normalization
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean_and_std, mean_and_std)],
        std=[1/s for s in mean_and_std]
    )
    
    # Apply unnormalization
    unnormalized_tensor = unnormalize(tensor_image)

    # Clip to [0, 1] just in case of any numerical overflow
    unnormalized_tensor = unnormalized_tensor.clamp(0, 1)

    # Convert back to PIL image
    img = to_pil_image(unnormalized_tensor)

    plt.imshow(img)
    plt.axis('off')  # hides the axis
    plt.show()


# Tab to agario
pyautogui.hotkey('alt', 'tab')
time.sleep(0.5)
pyautogui.keyDown('enter')
time.sleep(1)


frames = []
for _ in range(FRAMES_PER):
    agario_sc = pyautogui.screenshot(region=(0, 243, screen_width, 1277))
    frames.append(agario_sc)
    time.sleep(0.2)

plt.imshow(frames[0].convert('RGB'))
plt.axis('off')  # hides the axis
plt.show()

plt.imshow(frames[1].convert('RGB'))
plt.axis('off')  # hides the axis
plt.show()

frames = [frame.convert('L') for frame in frames]

print("Images")
plt.imshow(frames[0].convert('RGB'))
plt.axis('off')  # hides the axis
plt.show()

plt.imshow(frames[1].convert('RGB'))
plt.axis('off')  # hides the axis
plt.show()

# Stack them together
# Weird hack but use RGBA to do so
image_with_four_frames = Image.merge("LA", tuple(frames))

print("Merged")
plt.imshow(image_with_four_frames)
plt.axis('off')  # hides the axis
plt.show()

# Preprocess to be same dimensions
preprocessed_image = new_preprocess(image_with_four_frames)
print(preprocessed_image.shape)

print("Preprocessed")

show_preprocessed_image(preprocessed_image)

print("CNN")

batch = torch.stack([preprocessed_image]) 
state = torch.tensor(batch, dtype=torch.float32)
tester = DQN(n_frames=FRAMES_PER) # Theta 1
with torch.no_grad():
    cnn_output = tester(state)


print(cnn_output.shape)
