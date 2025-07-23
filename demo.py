import time
import gymnasium as gym
import panda_mujoco_gym

if __name__ == "__main__":

    # env = gym.make("FrankaShelfPNPSparse-v0", render_mode="human")
    env = gym.make("FrankaShelfPNPDense-v0", render_mode="human")

    observation, info = env.reset()
    print("ee初始位置坐标：", env.unwrapped.get_ee_position())

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.02)  # You can adjust the speed of the simulation by changing the time.sleep value

    env.close()