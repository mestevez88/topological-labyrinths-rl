import gym

import topological_labyrinths_rl


if __name__ == '__main__':
    env = gym.make("meta-topological-labyrinths-2D-v0")
    obs = env.reset()
    print(obs)
    done = False
    sum_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        sum_reward += reward
        env.render()
    print(f"Episode is over! You got {round(sum_reward, 2)} score.")
