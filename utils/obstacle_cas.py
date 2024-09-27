import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import platform
from multipledispatch import dispatch
import time
from palettable.colorbrewer.qualitative import Set1_9

# from envs.Obstacle.env_utils import *
from env_utils import *

class action_space_obj:
    def __init__(self):
        action_dim = 2
        self.low = (-0.5, -0.5)
        self.high = (0.5, 0.5)
        self.shape = (action_dim,)


class state_space_obj:
    def __init__(self, ENV_SIZE=10):
        state_dim = 2
        self.low = (0, 0)
        self.high = (ENV_SIZE, ENV_SIZE)
        self.shape = (state_dim,)


class init_space_obj:
    def __init__(self, ENV_SIZE=10):
        state_dim = 2
        self.low = (0, 0)
        self.high = (ENV_SIZE, ENV_SIZE)
        self.shape = (state_dim,)


class Obstacle:
    def __init__(self, gui, save_dir, params, horizon=50):
        self.cmap = Set1_9.mpl_colors
        self.num_agents = params['NUM_AGENTS']
        self.env_size = params['ENV_SIZE']
        self.obstacles = params['OBSTACLES']
        self.num_obstacles = len(self.obstacles)
        self.radii = params['RADII']
        self.goals = params['GOALS']
        self.dt = params['dt']
        self.num_goals = len(self.goals)
        self.state_space = state_space_obj(self.env_size)
        self.action_space = action_space_obj()
        self.init_space = init_space_obj(self.env_size)
        init = np.random.rand(self.num_agents, self.state_space.shape[0]) * self.env_size
        init = ensure_positions_outside_obstacles(init, self.obstacles, self.radii, self.state_space)
        self.agents = init
        self.horizon = horizon
        self.gui = gui
        self.save_dir = save_dir
        self.frames = []
        self.agents_history = [[self.agents[i].copy()] for i in range(self.num_agents)]  # History of agents' positions
        self.goal_idx = np.zeros(self.num_agents, dtype=int)

    def _get_obs(self):
        return self.agents.flatten()  # Flatten to return as 1D array

    @property
    def env_name(self):
        return "obstacle"

    @dispatch(int)
    def reset(self, episode):
        init = np.random.rand(self.num_agents, self.state_space.shape[0]) * self.env_size
        init = ensure_positions_outside_obstacles(init, self.obstacles, self.radii, self.state_space)
        self.agents = init

        self.goal_idx = np.zeros(self.num_agents, dtype=int)
        self.goal_states = np.zeros((self.num_agents, self.state_space.shape[0]))
        possibilities = list(range(self.num_goals))
        for i in range(self.num_agents):
            self.goal_idx[i] = np.random.choice(possibilities)
            self.goal_states[i] = self.goals[self.goal_idx[i], :]
            possibilities.remove(self.goal_idx[i])

        self.act_mat = [np.zeros(self.action_space.shape) for _ in range(self.num_agents)]

        self.episode = episode
        self.agents_history = [[self.agents[i].copy()] for i in range(self.num_agents)]  # Reset history on reset
        if self.gui:
            self.frames = []
            plt.close()
            self.fig, self.ax = plt.subplots()

        return self._get_obs()

    @dispatch(int, list)
    def reset(self, episode, info):
        init = np.random.rand(self.num_agents, self.state_space.shape[0]) * self.env_size
        init = ensure_positions_outside_obstacles(init, self.obstacles, self.radii, self.state_space)
        self.agents = init

        self.goal_idx = np.zeros(self.num_agents, dtype=int)
        self.goal_states = np.zeros((self.num_agents, self.state_space.shape[0]))
        possibilities = list(range(self.num_goals))
        for i in range(self.num_agents):
            self.goal_idx[i] = np.random.choice(possibilities)
            self.goal_states[i] = self.goals[self.goal_idx[i], :]
            possibilities.remove(self.goal_idx[i])

        self.act_mat = [np.zeros(self.action_space.shape) for _ in range(self.num_agents)]

        self.episode = episode
        self.agents_history = [[self.agents[i].copy()] for i in range(self.num_agents)]  # Reset history on reset
        if self.gui:
            self.frames = []
            plt.close()
            self.fig, self.ax = plt.subplots()

        return self._get_obs()

    def agent_policy(self):
        states = self.agents
        goals = self.goal_states
        opti, X, X_ref, U_ref, num_agents, state_dim, control_dim, prediction_horizon = multi_agent_rhc(
            states, goals, self.obstacles, self.radii, self.state_space, self.action_space, self.horizon, self.dt
        )

        res, _ = solve_opti(opti, X, X_ref, U_ref, goals, num_agents, state_dim, control_dim, prediction_horizon)
        if res is None:
            return False
        else:
            for i in range(num_agents):
                self.act_mat[i] = res[i, :] * self.dt
            return True

    def step(self):
        valid_result = self.agent_policy()
        if not valid_result:
            for i in range(self.num_agents):
                self.act_mat[i] = np.random.uniform(self.action_space.low, self.action_space.high)
        for i in range(self.num_agents):
            self.agents[i] = np.clip(self.agents[i] + self.act_mat[i], self.state_space.low, self.state_space.high)

        done = False

        for i in range(self.num_agents):
            self.agents_history[i].append(self.agents[i].copy())

        if self.gui:
            self.render()

        dist_to_goal = np.linalg.norm(self.agents - self.goal_states, axis=1)
        if np.all(dist_to_goal < 0.05 * self.env_size):
            done = True
            plt.close()
        return self._get_obs(), done, np.array(self.act_mat)

    def render(self):
        self.ax.clear()

        self.ax.vlines([self.state_space.low[0], self.state_space.high[0]], self.state_space.low[1], self.state_space.high[1], 'k')
        self.ax.hlines([self.state_space.low[1], self.state_space.high[1]], self.state_space.low[0], self.state_space.high[0], 'k')
        self.ax.axis('equal')
        self.ax.set_xlim(self.state_space.low[0], self.state_space.high[0])
        self.ax.set_ylim(self.state_space.low[1], self.state_space.high[1])

        # goals
        self.ax.scatter(self.goals[self.goal_idx, 0], self.goals[self.goal_idx, 1], marker='*', s=30, color='g')
        for i in range(self.num_agents):
            a_circle = plt.Circle((self.goal_states[i, 0], self.goal_states[i, 1]), 0.2, color=self.cmap[i], fill=False)
            self.ax.add_patch(a_circle)

        # obstacles
        for idx in range(self.num_obstacles):
            obstacle = plt.Circle((self.obstacles[idx, 0], self.obstacles[idx, 1]), radius=self.radii[idx], color='gray', fill=True)
            self.ax.add_patch(obstacle)

        # agents and their history
        s = self.agents  # agents are already in a 2D array
        for i in range(self.num_agents):
            self.ax.plot(s[i, 0], s[i, 1], c=self.cmap[i], marker='.')
            # Plot the history of each agent as a trajectory
            history = np.array(self.agents_history[i].copy())
            if len(history) > 1:
                self.ax.plot(history[:, 0], history[:, 1], c=self.cmap[i], linestyle='--')

        self.fig.canvas.draw()
        plt.pause(0.1)

    def save_gui(self):
        if self.frames:
            os.makedirs(f'{self.save_dir}/gui_obstacle', exist_ok=True)
            with imageio.get_writer(f'{self.save_dir}/gui_obstacle/{self.episode}.gif', mode='I', duration=0.5) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
