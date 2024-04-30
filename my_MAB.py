import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        Receives the parameters that are available to the algorithm and builds an object of the class
        """
        # TODO: Decide what/if to store. Could be used in the future
        # Store the necessary parameters
        self.num_arms = num_arms
        self.num_users = num_users
        self.phase_len = phase_len
        self.arms_thresh = arms_thresh
        self.users_distribution = users_distribution
        self.num_rounds = num_rounds
        self.exploration_par = ((self.num_rounds / self.num_arms) * (2 / 3)) * (np.log(self.num_rounds)) ** (1 / 3)
        self.exploration_rounds = self.exploration_par * self.num_arms
        self.exploration_phases_num = self.exploration_rounds / self.phase_len
        self.explore_phase = 0

        # Initialize variables for tracking user statistics
        self.current_user = 0

        # Initialize variables for tracking arm statistics
        self.chosen_arm = 0
        self.active_arms = np.arange(self.num_arms)
        self.arm_counts_for_user = np.zeros((num_arms, num_users))
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards_for_user = np.zeros((num_arms, num_users))
        self.arm_counts_per_phase = np.zeros(num_arms)

        self.total_rounds = 0
        self.count_rounds_per_phase = 0

        # Maximum likelihood estimation parameters
        self.MLE_user_rewards_from_arms = np.zeros((num_users, num_arms))



    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        Receives the parameters that are available to the algorithm and builds an object of the class
        """
        # TODO: This is your place to shine. Go crazy!
        self.count_rounds_per_phase += 1
        self.current_user = user_context
        self.total_rounds += 1
        # Explore arms that have not reached the threshold
        if self.explore_phase <= 99.9*self.phase_len*(1-((max(self.users_distribution) - min(self.users_distribution))/(max(self.users_distribution) + min(self.users_distribution)))):
            for arm in self.active_arms:
                # I want to keep the arm active to explore it more by choosing it as long as it didn't hit the thresh
                if self.phase_len - self.count_rounds_per_phase == self.arms_thresh[arm] - self.arm_counts_per_phase[arm]:
                    self.chosen_arm = arm
                    self.arm_counts_for_user[self.chosen_arm][user_context] += 1
                    self.arm_counts_per_phase[self.chosen_arm] += 1
                    self.arm_counts[self.chosen_arm] += 1
                    return self.chosen_arm

        # Implement the UCB algorithm for arm selection
        exploration_factor = 2  # Exploration parameter
        ucb_values = np.zeros(self.num_arms)

        for arm in self.active_arms:
            if self.arm_counts[arm] == 0 or self.arm_counts_for_user[arm][user_context] == 0:
                ucb_values[arm] = float('inf')
            else:
                avg_reward = self.arm_rewards_for_user[arm][user_context] / (self.arm_counts_for_user[arm][user_context])
                exploration_bonus = np.sqrt(exploration_factor * np.log(self.total_rounds) / (self.arm_counts[arm]))
                ucb_values[arm] = avg_reward + exploration_bonus
        if self.total_rounds < 900000:
            self.chosen_arm = np.argmax(self.MLE_user_rewards_from_arms[user_context])
        else:
            self.chosen_arm = np.argmax(ucb_values)

        self.arm_counts_for_user[self.chosen_arm][user_context] += 1
        self.arm_counts_per_phase[self.chosen_arm] += 1
        self.arm_counts[self.chosen_arm] += 1
        return self.chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        A function that is called at the end of a round and receives:
reward o – the profit of the user from the arm recommended to him.
        """
        # TODO: Use this information for your algorithm
        # Calculate the modified reward based on the arms_thresh and users_distribution
        user_probability = self.users_distribution[self.current_user]
        reward_parameter = self.arms_thresh[self.chosen_arm]/self.phase_len
        modified_reward = reward * reward_parameter / user_probability

        self.arm_rewards_for_user[self.chosen_arm][self.current_user] += modified_reward

        # MLE CALC
        if reward > self.MLE_user_rewards_from_arms[self.current_user][self.chosen_arm]:
            self.MLE_user_rewards_from_arms[self.current_user][self.chosen_arm] = reward

        # Don't use deactivated arms
        if self.count_rounds_per_phase == self.phase_len:

            self.explore_phase += 1
            self.count_rounds_per_phase = 0

            for arm in self.active_arms:
                if self.arm_counts_per_phase[arm] < self.arms_thresh[arm]:
                    self.active_arms = self.active_arms[self.active_arms != arm]
            self.arm_counts_per_phase = self.arm_counts_per_phase * 0

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id"
