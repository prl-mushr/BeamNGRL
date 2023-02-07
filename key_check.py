import time
import numpy as np
from pynput import keyboard, mouse
from pynput.mouse import Controller
# from screeninfo import get_monitors #uncomment if using python>=3.6

comp_mouse = Controller()

# monitors = get_monitors()
# monitors = monitors[0]

screen_width = 1960#monitors.width
screen_height = 1080#monitors.height

mouse_x = screen_width//2
mouse_y = screen_height//2

pressed_key=''

def on_press(key):
    global pressed_key
    try:
        pressed_key = key.char
    except AttributeError:
        pass
        # print('special key {0} pressed'.format(
        #     key))

def on_release(key):
    global pressed_key
    pressed_key = ''
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
listener = keyboard.Listener(on_press=on_press,on_release=on_release)
listener.start()


def on_move(x, y):
    global mouse_x
    global mouse_y
    mouse_x = x
    mouse_y = y

listener_mouse = mouse.Listener(on_move=on_move)
listener_mouse.start()

KeyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KeyList.append(char)
    
def key_press():

    global pressed_key
    return [str(pressed_key).upper()]

def reset_mouse_pos():
    global screen_width
    global screen_height
    comp_mouse.position = (screen_width//2,screen_height//2)

def get_mouse_pos():
    global screen_width
    global screen_height
    global mouse_x
    global mouse_y
    x,y = mouse_x, mouse_y
    x -= screen_width//2
    x /= (screen_width)*0.25
    y -= screen_height//2
    y /= -(screen_height)*0.25
    return x,y

