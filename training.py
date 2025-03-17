import directkeys
from grab_screen import grab_screen
from DQN import DQNAgent
import cv2
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from screeninfo import get_monitors
import keyboard  # For detecting key presses
import torch
from PIL import Image

window_size = (0,0,1280,720)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#monitor = get_monitors()[0]

#window_size = (0, 0, monitor.width, monitor.height)

# 定义动作
action_texts = {
    0: "Use main skill",
    1: "Use secondary skill",
    2: "Use third skill",
    3: "Dodge the attack",
    4: "Move left",
    5: "Move up",
    6: "Move down",
    7: "Move right",
    8: "pickup",
    9: "interact"
}

action_texts_list = list(action_texts.values())

def take_action(action):
    action_mapping = {
        0: directkeys.mainskill,
        1: directkeys.secondskill,
        2: directkeys.thirdskill,
        3: directkeys.dodge,
        4: directkeys.move_left,
        5: directkeys.move_up,
        6: directkeys.move_down,
        7: directkeys.move_right,
        8: directkeys.pickup,
        9: directkeys.interact
    }
    if action in action_mapping:
        action_mapping[action]()
    return action_texts.get(action, "Unknown action")


# 训练参数
action_size = len(action_texts)
print(action_size)
WIDTH, HEIGHT = 96, 88
num_episodes = 3000
update_step = 50
batch_size = 16
DQN_model_path = "model_gpu"
DQN_log_path = "log_path"


def preprocess_screen(original_screen):
    original_screen = cv2.resize(original_screen, (WIDTH, HEIGHT))
    original_screen = cv2.cvtColor(original_screen, cv2.COLOR_BGR2RGB)
    return Image.fromarray(original_screen)


def emergency_stop(episode):
    # Save the model and replay buffer in case of emergency
    print(f"Emergency! Saving model and replay buffer at episode {episode}")
    agent.save_model(episode)

    print("Program paused. Press the 'End' key to resume.")
    while True:
        if keyboard.is_pressed("end"):
            print("Resuming program...")
            break  # Break out of the loop and continue execution



def check_for_emergency():
    # Check if the "End" key is pressed
    if keyboard.is_pressed("end"):
        return True
    return False


if __name__ == '__main__':
    agent = DQNAgent(action_size, action_texts_list)

    for episode in range(num_episodes):
        screen = grab_screen(window_size)
        state = preprocess_screen(screen)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Check for emergency stop (if "End" key is pressed)
            if check_for_emergency():
                emergency_stop(episode)

            start_time = time.time()
            action, _, reward = agent.select_action(state, action_texts_list)
            take_action(action)

            next_screen = grab_screen(window_size)
            next_state = preprocess_screen(next_screen)

            # 计算外部奖励
            #external_reward = compute_external_rewards()

            agent.store_data(state, action_texts_list, reward, next_state)

            if len(agent.replay_buffer) > batch_size:
                agent.train(batch_size, step_count)
            if step_count % update_step == 0:
                agent.update_target_net()

            state = next_state
            step_count += 1
            print(f'Step {step_count} took {time.time() - start_time:.4f} seconds, reward:{reward}')

        if episode % 10 == 0:
            agent.save_model(episode)

        print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {step_count}")










