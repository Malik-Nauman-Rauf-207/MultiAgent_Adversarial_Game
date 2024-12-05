import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, value, PULP_CBC_CMD
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg



class ZeroSumGridWorld:
    def __init__(self, grid_size: int = 4):
        """
        Initialize Zero-Sum Grid World environment with Min-Max Q-learning

        :param grid_size: Size of the grid world
        """
        self.grid_size = grid_size

        # Possible actions (8 directions)
        self.actions = [
            'N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW'
        ]

        # Action mapping for uncertainty dynamics
        self.action_map = {
            'N': ['NW', 'NE'],
            'E': ['NE', 'SE'],
            'W': ['NW', 'SW'],
            'S': ['SE', 'SW'],
            'NE': ['N', 'E'],
            'NW': ['N', 'W'],
            'SE': ['S', 'E'],
            'SW': ['S', 'W']
        }

        # Q-values for both agents
        self.Q1 = {}  # Protagonist (minimizer)
        self.Q2 = {}  # Adversary (maximizer)

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon_start = 1.0  # Start with full exploration
        self.epsilon_end = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.99  # Decay rate for exploration
        self.current_episode = 0

        # Goal and forbidden regions
        self.goal_region = self._define_goal_region()
        self.forbidden_region = self._define_forbidden_region()

    def _define_goal_region(self) -> List[Tuple[int, int]]:
        """Define goal region"""
        return [(x, y) for x in range(1) for y in range(2, 4)]

    def _define_forbidden_region(self) -> List[Tuple[int, int]]:
        """Define forbidden region (to be avoided)"""
        return [(x, y) for x in range(1, 2) for y in range(2, 3)]

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is within grid boundaries

        :param state: Current state coordinates
        :return: Boolean indicating state validity
        """
        x, y = state
        return (0 <= x < self.grid_size) and (0 <= y < self.grid_size)

    def execute_action(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        Execute an action with uncertainty dynamics

        :param state: Current state
        :param action: Chosen action
        :return: New state after action
        """
        # 95% chance of executing selected action, 5% chance of nearby actions
        if random.random() < 0.95:
            dx, dy = self._get_action_delta(action)
        else:
            alt_action = random.choice(self.action_map[action])
            dx, dy = self._get_action_delta(alt_action)

        new_x, new_y = state[0] + dx, state[1] + dy

        # Boundary checking
        if not self.is_valid_state((new_x, new_y)):
            return state

        return (new_x, new_y)

    def _get_action_delta(self, action: str) -> Tuple[int, int]:
        """
        Get coordinate deltas for each action

        :param action: Action string
        :return: Coordinate change (dx, dy)
        """
        action_deltas = {
            'N': (0, -1), 'S': (0, 1),
            'E': (1, 0), 'W': (-1, 0),
            'NE': (1, -1), 'NW': (-1, -1),
            'SE': (1, 1), 'SW': (-1, 1)
        }
        return action_deltas[action]

    def compute_reward(self, state1: Tuple[int, int], state2: Tuple[int, int], timestep: int) -> Tuple[float, float]:
        """
        Compute rewards for both agents in a zero-sum manner

        :param state1: Protagonist agent state
        :param state2: Adversary agent state
        :param timestep: Current timestep
        :return: Rewards for agent 1 and agent 2
        """
        # Protagonist (Minimizer) reward
        if state1 in self.goal_region and state1 not in self.forbidden_region:
            # Reached goal within 5 steps
            if timestep <= 10:
                return 1, -1  # Note the reversed signs for zero-sum

        # Adversary (Maximizer) prevents goal
        if state1 in self.forbidden_region or state2 == state1:
            return -1, 1.0  # Note the reversed signs for zero-sum

        # Neutral state
        return 0.0, 0.0

    def get_epsilon(self):
        """
        Compute epsilon value with exponential decay

        :return: Current exploration rate
        """
        # Exponential decay of epsilon
        epsilon = max(
            self.epsilon_end,
            self.epsilon_start * (self.epsilon_decay ** self.current_episode)
        )
        return epsilon

    def select_action_p(self, state: Tuple[int, int], Q_values: Dict) -> str:
        """
        Select action using epsilon-greedy strategy

        :param state: Current state
        :param Q_values: Q-values dictionary
        :return: Selected action
        """
        # Get current exploration rate
        epsilon = self.get_epsilon()

        # Exploration: random action
        if random.random() < epsilon:
            return random.choice(self.actions)

        # Exploitation: select action with max Q-value
        state_actions = [
            Q_values.get((state, action), 0.0) for action in self.actions
        ]
        max_q_value = max(state_actions)
        best_actions = [
            action for action, q_value in zip(self.actions, state_actions)
            if q_value == max_q_value
        ]

        return random.choice(best_actions)

    def select_action_a(self, state: Tuple[int, int], Q_values: Dict) -> str:
        """
        Select action using epsilon-greedy strategy

        :param state: Current state
        :param Q_values: Q-values dictionary
        :return: Selected action
        """
        # Get current exploration rate
        epsilon = self.get_epsilon()

        # Exploration: random action
        if random.random() < epsilon:
            return random.choice(self.actions)

        # Exploitation: select action with max Q-value
        state_actions = [
            Q_values.get((state, action), 0.0) for action in self.actions
        ]
        min_q_value = min(state_actions)
        best_actions = [
            action for action, q_value in zip(self.actions, state_actions)
            if q_value == min_q_value
        ]

    # Randomly choose among best actions if multiple exist
        return random.choice(best_actions)

    def minimax_q_update(self, Q1: Dict, Q2: Dict, state1: Tuple[int, int], state2: Tuple[int, int],
                         action1: str, action2: str, next_state1: Tuple[int, int],
                         next_state2: Tuple[int, int], reward1: float, reward2: float):
        """
        Perform True Minimax Q-learning update for both agents

        :param Q1: Q-values for protagonist (minimizer)
        :param Q2: Q-values for adversary (maximizer)
        :param state1: Current state of protagonist
        :param state2: Current state of adversary
        :param action1: Action chosen by protagonist
        :param action2: Action chosen by adversary
        :param next_state1: Next state of protagonist
        :param next_state2: Next state of adversary
        :param reward1: Reward for protagonist
        :param reward2: Reward for adversary
        """
        # Protagonist Update: min(max(Q-values))
        old_q1 = Q1.get((state1, action1), 0.0)

        # For each possible protagonist action, find the max Q-value from adversary's perspective
        max_q2_for_each_action = []
        for next_protag_action in self.actions:
            # Find worst Q-values for this protagonist action from adversary's perspective
            q2_values_for_action = [
                Q2.get((next_state1, next_adv_action), 0.0)
                for next_adv_action in self.actions
            ]
            # Take max from adversary's perspective
            max_q2_for_action = min(q2_values_for_action)
            max_q2_for_each_action.append(max_q2_for_action)

        # Take min over max Q-values (protagonist's perspective)
        next_q1 = max(max_q2_for_each_action)

        # Update protagonist's Q-value
        new_q1 = (1 - self.learning_rate) * old_q1 + \
                 self.learning_rate * (reward1 + self.discount_factor * next_q1)
        Q1[(state1, action1)] = new_q1


        # Adversary Update: max(min(Q-values))
        old_q2 = Q2.get((state2, action2), 0.0)

        # For each possible adversary action, find the min Q-value from protagonist's perspective
        min_q1_for_each_action = []
        for next_adv_action in self.actions:
            # Find worst Q-values for this adversary action from protagonist's perspective
            q1_values_for_action = [
                Q1.get((next_state2, next_protag_action), 0.0)
                for next_protag_action in self.actions
            ]
            # Take min from protagonist's perspective
            min_q1_for_action = max(q1_values_for_action)
            min_q1_for_each_action.append(min_q1_for_action)

        # Take max over min Q-values (adversary's perspective)
        next_q2 = min(min_q1_for_each_action)

        # Update adversary's Q-value
        new_q2 = (1 - self.learning_rate) * old_q2 + \
                 self.learning_rate * (reward2 + self.discount_factor * next_q2)
        Q2[(state2, action2)] = new_q2

    # def update_policy(self, is_protagonist: bool = True):
    #     """
    #     Update policy using linear programming for a specific agent
    #
    #     :param is_protagonist: True for protagonist (minimizer), False for adversary (maximizer)
    #     """
    #     # Prepare policy dictionary if not exists
    #     if not hasattr(self, 'policy'):
    #         self.policy = {}
    #
    #     # Iterate through all possible states
    #     for state in [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]:
    #         # State key for policy indexing
    #         state_key = state
    #
    #         # Create LP problem
    #         prob = LpProblem("MiniMaxQPolicy", LpMaximize if is_protagonist else LpMinimize)
    #
    #         # Decision variables
    #         v = LpVariable("v")
    #         pi = LpVariable.dicts("pi", self.actions, lowBound=0, upBound=1)
    #
    #         # Objective function
    #         prob += v
    #
    #         # Constraint: Probability distribution sums to 1
    #         prob += lpSum(pi.values()) == 1
    #
    #         if is_protagonist:
    #             # Protagonist's policy update (minimization)
    #             for opp_action in self.actions:
    #                 # Compute expected Q-values over opponent's actions
    #                 expr = lpSum(
    #                     pi[action] * min(
    #                         self.Q2.get((state, opp_action), 0.0)
    #                         for opp_action in self.actions
    #                     )
    #                     for action in self.actions
    #                 )
    #                 prob += expr >= v
    #         else:
    #             # Adversary's policy update (maximization)
    #             for protag_action in self.actions:
    #                 # Compute expected Q-values over protagonist's actions
    #                 expr = lpSum(
    #                     pi[action] * max(
    #                         self.Q1.get((state, protag_action), 0.0)
    #                         for protag_action in self.actions
    #                     )
    #                     for action in self.actions
    #                 )
    #                 prob += expr <= v
    #
    #         # Solve the linear programming problem
    #         prob.solve(PULP_CBC_CMD(msg=0))
    #
    #         # Store the policy for this state
    #         self.policy[state_key] = {
    #             action: value(pi[action]) for action in self.actions
    #         }
    #
    # def print_policy_details(self):
    #     """
    #     Print detailed policy probabilities
    #     """
    #     if not hasattr(self, 'policy'):
    #         print("No policy has been computed yet.")
    #         return
    #
    #     print("\nDetailed Policy Probabilities:")
    #     for state, action_probs in sorted(self.policy.items()):
    #         print(f"\nState {state}:")
    #         for action, prob in action_probs.items():
    #             print(f"  {action}: {prob:.4f}")

    def train(self, num_episodes: int = 50000):
        """
        Train agents using Minimax Q-learning

        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            # Track current episode for epsilon decay
            self.current_episode = episode

            # Initial states
            state1 = (self.grid_size - 4, 0) # Protagonist starts
            state2 = (self.grid_size - 2, self.grid_size - 2)  # Adversary starts opposite corner

            for timestep in range(10):  # Max 7 steps per episode
                # Select actions with epsilon-greedy
                action1 = self.select_action_p(state1, self.Q1)
                action2 = self.select_action_a(state2, self.Q2)

                # Execute actions
                next_state1 = self.execute_action(state1, action1)
                next_state2 = self.execute_action(state2, action2)

                # Compute rewards
                reward1, reward2 = self.compute_reward(next_state1, next_state2, timestep)

                # True Minimax Q-value update
                self.minimax_q_update(
                    self.Q1, self.Q2,
                    state1, state2,
                    action1, action2,
                    next_state1, next_state2,
                    reward1, reward2
                )

                # Update states
                state1, state2 = next_state1, next_state2

                # Check for terminal conditions
                if reward1 == -1.0 or reward2 == -1.0:
                    break

            if (episode + 1) % 100000 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")

    def visualize_trajectory(self):
        """
        Visualize the agents' trajectories using learned policies
        """
        plt.figure(figsize=(10, 10))
        plt.title("Minimax Zero-Sum Grid World Trajectory")
        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(self.grid_size - 0.5, -0.5)
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Plot grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                plt.text(x, y, f'({x},{y})', ha='center', va='center', alpha=0.5)

        # Plot goal and forbidden regions
        for x, y in self.goal_region:
            plt.fill([x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                     [y - 0.5, y - 0.5, y + 0.5, y + 0.5],
                     color='green', alpha=0.3)
        for x, y in self.forbidden_region:
            plt.fill([x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                     [y - 0.5, y - 0.5, y + 0.5, y + 0.5],
                     color='red', alpha=0.3)

        # Trajectories
        state1 = (self.grid_size - 4, 0)  # Protagonist starts
        state2 = (self.grid_size - 2, self.grid_size - 2)
        trajectory1 = [state1]
        trajectory2 = [state2]

        for _ in range(10):  # Max 7 steps
            # Select best actions with no exploration
            action1 = self.select_action_p(state1, self.Q1)
            action2 = self.select_action_a(state2, self.Q2)

            # Execute actions
            next_state1 = self.execute_action(state1, action1)
            next_state2 = self.execute_action(state2, action2)

            trajectory1.append(next_state1)
            trajectory2.append(next_state2)

            # Check for terminal conditions
            reward1, reward2 = self.compute_reward(next_state1, next_state2, _)
            if reward1 == -1.0 or reward2 == -1.0:
                break

            state1, state2 = next_state1, next_state2

        # Plot trajectories
        traj1_x, traj1_y = zip(*trajectory1)
        traj2_x, traj2_y = zip(*trajectory2)

        plt.plot(traj1_x, traj1_y, marker='o', linestyle='-', color='blue',
                 linewidth=2, markersize=10, label='Protagonist (Maximizer)')
        plt.plot(traj2_x, traj2_y, marker='s', linestyle='-', color='red',
                 linewidth=2, markersize=10, label='Adversary (Minimizer)')

        # Annotate trajectories
        for i, (x, y) in enumerate(trajectory1):
            plt.annotate(str(i), (x, y), xytext=(5, 5),
                         textcoords='offset points', fontweight='bold', color='blue')
        for i, (x, y) in enumerate(trajectory2):
            plt.annotate(str(i), (x, y), xytext=(-15, -15),
                         textcoords='offset points', fontweight='bold', color='red')

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.tight_layout()
        plt.savefig('trajectory_plot.jpg', dpi=300, bbox_inches='tight')
        plt.show()


    def evaluate_policy(self, num_trials: int = 100):
        """
        Evaluate learned policies with no exploration

        :param num_trials: Number of evaluation trials
        :return: Success rate for protagonist
        """
        success_count = 0

        for _ in range(num_trials):
            state1 = (self.grid_size - 4, 0)  # Protagonist starts
            state2 = (self.grid_size - 2, self.grid_size - 2)

            for timestep in range(10):
                # Select best actions with zero exploration
                action1 = self.select_action_p(state1, self.Q1)
                action2 = self.select_action_a(state2, self.Q2)

                # Execute actions
                next_state1 = self.execute_action(state1, action1)
                next_state2 = self.execute_action(state2, action2)

                # Check for terminal conditions
                reward1, reward2 = self.compute_reward(next_state1, next_state2, timestep)

                if reward1 == -1.0:
                    success_count += 1
                    break

                state1, state2 = next_state1, next_state2

        return success_count / num_trials

    def visualize_agents_with_icons(self, protagonist_icon_path, adversary_icon_path):
        """
        Visualize the agents' positions using icons for the protagonist and adversary.

        Parameters:
        - protagonist_icon_path: Path to the icon image for the protagonist.
        - adversary_icon_path: Path to the icon image for the adversary.
        """
        plt.figure(figsize=(10, 10))
        plt.title("Minimax Zero-Sum Grid World")
        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(self.grid_size - 0.5, -0.5)
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Plot grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                plt.text(x, y, f'({x},{y})', ha='center', va='center', alpha=0.5)

        # Plot goal and forbidden regions
        for x, y in self.goal_region:
            plt.fill([x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                     [y - 0.5, y - 0.5, y + 0.5, y + 0.5],
                     color='green', alpha=0.3)
        for x, y in self.forbidden_region:
            plt.fill([x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                     [y - 0.5, y - 0.5, y + 0.5, y + 0.5],
                     color='red', alpha=0.3)

        # Starting positions
        protagonist_state = (self.grid_size - 4, 0)  # Protagonist
        adversary_state = (self.grid_size - 2, self.grid_size - 2)  # Adversary

        # Load icons
        protagonist_icon = mpimg.imread(protagonist_icon_path)
        adversary_icon = mpimg.imread(adversary_icon_path)

        # Function to place an icon at a specific position
        def add_icon(x, y, icon, ax, zoom=0.2):
            img = OffsetImage(icon, zoom=zoom)
            ab = AnnotationBbox(img, (x, y), frameon=False)
            ax.add_artist(ab)

        ax = plt.gca()
        add_icon(*protagonist_state, protagonist_icon, ax)
        add_icon(*adversary_state, adversary_icon, ax)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.tight_layout()
        plt.savefig('agents_plot.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    # Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create and train zero-sum game agents
    zero_sum_game = ZeroSumGridWorld(grid_size=4)
    zero_sum_game.train(num_episodes=500000)

    # Evaluate policy
    success_rate = zero_sum_game.evaluate_policy()
    print(f"Protagonist Success Rate: {success_rate * 100:.2f}%")

    # Visualize trajectory
    zero_sum_game.visualize_trajectory()

    zero_sum_game.visualize_agents_with_icons('Protagonist_Icon.png', 'Adversary_Icon.png')