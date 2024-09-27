import numpy as np
import casadi as ca
import time


def ensure_positions_outside_obstacles(positions, obstacles, radii, state_space, buffer_distance=0.1):
    for i in range(len(positions)):
        valid_position = False
        while not valid_position:
            d = np.linalg.norm(positions[i] - obstacles, axis=1)
            if np.any([d[ii] < radii[ii] + buffer_distance for ii in range(len(radii))]):
                positions[i] = np.random.uniform(state_space.low, state_space.high)
            else:
                valid_position = True
    return positions

def set_initial_guess(opti, X, U, initial_states, reference_states, prediction_horizon, time_step, max_velocity=1.0):
    num_agents = initial_states.shape[0]
    state_dim = initial_states.shape[1]

    for i in range(num_agents):
        # Direction vector from initial state to reference (goal) state
        direction = reference_states[i] - initial_states[i]
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-6:  # Avoid division by zero
            direction /= direction_norm
        
        # Assume constant velocity along the line toward the goal
        velocity = min(max_velocity, direction_norm / (prediction_horizon * time_step))
        velocity_vector = velocity * direction
        
        # Set the initial guess for positions (X) and velocities (U)
        for k in range(prediction_horizon + 1):
            # Linearly interpolate positions towards the goal
            position_guess = initial_states[i] + k * time_step * velocity_vector
            opti.set_initial(X[i][k], position_guess)
            
            if k < prediction_horizon:
                # Set the initial control input to maintain the velocity towards the goal
                opti.set_initial(U[i][k], velocity_vector)

def multi_agent_rhc(initial_states, reference_states, obstacles, radii, state_space, action_space, prediction_horizon=10, time_step=0.1, slack_penalty=1e7):
    num_agents = initial_states.shape[0]
    state_dim = initial_states.shape[1]
    control_dim = state_dim  # Assuming control dimension matches state dimension for simplicity

    def system_dynamics(x, u):
        return x + u * time_step
    
    min_state = state_space.low
    max_state = state_space.high

    def calculate_goal_cost(x, x_ref, u, u_ref):
        distance_error = ca.sumsqr(x - x_ref)
        control_effort = ca.sumsqr(u - u_ref)
        # state_low = np.ones(x_ref.shape) * min_state[0]
        # state_high = np.ones(x_ref.shape) * max_state[0]
        # boundary_error = ca.sum1(state_low - x)
        # boundary_error += ca.sum1(x - state_high)
        return 5 * distance_error + 0.01 * control_effort# + 5 * boundary_error  # 0.01 is a weighting factor
    
    opti = ca.Opti()
    X = [[opti.variable(state_dim) for _ in range(prediction_horizon + 1)] for _ in range(num_agents)]
    U = [[opti.variable(state_dim) for _ in range(prediction_horizon)] for _ in range(num_agents)]
    slack = opti.variable(1)
    X_ref = opti.parameter(state_dim, num_agents)
    U_ref = np.zeros((state_dim, num_agents))

    total_cost = 0
    max_velocity = action_space.high
    min_velocity = action_space.low
    min_distance = 0.05

    for i in range(num_agents):
        opti.subject_to(X[i][0] == initial_states[i])
        for k in range(prediction_horizon):
            opti.subject_to(X[i][k + 1] == system_dynamics(X[i][k], U[i][k]))
            prediction_error = calculate_goal_cost(X[i][k], X_ref[:, i], U[i][k], U_ref[:, i])
            total_cost += prediction_error
            for dim in range(state_dim):
                opti.subject_to(U[i][k][dim] <= max_velocity[dim])
                opti.subject_to(U[i][k][dim] >= min_velocity[dim])
                opti.subject_to(X[i][k][dim] <= max_state[dim])
                opti.subject_to(X[i][k][dim] >= min_state[dim])
    # Collision avoidance constraints for other agents and obstacles
    for k in range(prediction_horizon):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = ca.sqrt(ca.sumsqr(X[i][k] - X[j][k]))
                # collision_avoidance_cost = ca.fmax(0, min_distance - distance) ** 2
                # total_cost += 1e6 * collision_avoidance_cost  # High penalty for collisions
                opti.subject_to(distance + slack >= min_distance)
            for o in range(len(obstacles)):
                distance_to_obstacle = ca.norm_2(X[i][k] - obstacles[o])
                opti.subject_to(distance_to_obstacle +  slack >= radii[o])
    opti.subject_to(slack >= 0)
    opti.minimize(total_cost + slack*slack_penalty)

    # Initialize variables
    # Apply the initial guess strategy
    set_initial_guess(opti, X, U, initial_states, reference_states, prediction_horizon, time_step)

    p_opts = {"verbose": False, 'print_time': False}
    s_opts = {
        "max_iter": 10000,
        "print_level": 0,
        # "tol": 1e-3,  # Slightly increase solver tolerance for speed
        # "acceptable_tol": 1e-2,  # Specify acceptable tolerance to allow early stopping
        # "acceptable_iter": 15  # Allow early stopping after a certain number of acceptable iterations
    }
    opti.solver('ipopt', p_opts, s_opts)
    # opti.solver('ipopt')
    return opti, X, X_ref, U, num_agents, state_dim, control_dim, prediction_horizon

def solve_opti(opti, X, X_ref, U, reference_states, num_agents, state_dim, control_dim, prediction_horizon):

    opti.set_value(X_ref, reference_states.T)
    
    # try:
    sol = opti.solve()
    # except RuntimeError as e:
    #     print(f"Solver failed: {e}")
    #     return None, None

    optimal_u = np.zeros((num_agents, control_dim))
    for i in range(num_agents):
        for dim in range(control_dim):
            optimal_u[i][dim] = sol.value(U[i][0][dim])

    trajectories = np.zeros((state_dim, prediction_horizon + 1, num_agents))
    for i in range(num_agents):
        for k in range(prediction_horizon + 1):
            for dim in range(state_dim):
                trajectories[dim, k, i] = sol.value(X[i][k][dim])

    return optimal_u, trajectories