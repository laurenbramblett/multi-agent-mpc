import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg', depending on your system

sys.path.append('utils')
import obstacle_cas as obs

if __name__ == '__main__':
    print("Running minimal_example.py")
    ENV_SIZE = 3
    dt = 0.3
    OBSTACLES = np.array([[0.27, 0.68],
                        [0.73, 0.73],
                        [0.61, 0.25]]) * ENV_SIZE
    # OBSTACLES = np.array([[1.9,0.7],[0.9,2.0],[2.2,2.2]])
    NUM_OBSTACLES = len(OBSTACLES)
    RADII = np.array([0.1, 0.18, 0.13]) * ENV_SIZE
    # RADII = np.array([0.2,0.3,0.3])
    GOALS = np.array([[0.215, 0.32],
                    [0.87, 0.93],
                    [0.04, 0.92],
                    [0.92, 0.07]]) * ENV_SIZE
    NUM_AGENTS = 2

    NUM_GOALS = len(GOALS)
    params = {
        'NUM_AGENTS': NUM_AGENTS,
        'ENV_SIZE': ENV_SIZE,
        'OBSTACLES': OBSTACLES,
        'RADII': RADII,
        'GOALS': GOALS,
        'dt': dt
        }

    if NUM_AGENTS > NUM_GOALS:
        raise ValueError("Number of agents should be less than or equal to the number of goals")

    env = obs.Obstacle(True,"",params,10)
    env.reset(1)
    iters = 0
    done = False
    while iters < 40 and not done:
        _, done, _ = env.step()
        iters += 1
        print("Locations: ", env.agents)
        print("Goals: ", env.goal_states)
        print("Rewards: ", [np.linalg.norm(env.agents[i] - env.goal_states[i]) for i in range(env.num_agents)])

    # env.render()
    # env.save_gui()