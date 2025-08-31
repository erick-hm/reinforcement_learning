import numpy as np


class GaussianBandit:
    """
    A Gaussian Bandit used to play a bandit game, which returns a
    random reward sampled from the normal distribution N(mean, std).
    """

    def __init__(self, mean: float = 0, std: float = 1):
        self.mean = mean
        self.std = std

    def pull_lever(self):
        """Generate a random reward using the specified normal distribution."""
        reward = np.random.normal(self.mean, self.std)
        return np.round(reward, 1)


class BernoulliBandit:
    """
    A Bernoulli Bandit used to play a bandit game, which returns a
    random reward sampled from the Bernoulli distribution Bernoulli(p).
    """

    def __init__(self, p: float):
        if p < 0 or p > 1:
            msg = "Probability p must be in the range [0,1]."
            raise ValueError(msg)

        self.p = p

    def pull_lever(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward


class GaussianBanditGame:
    """
    The Gaussian Bandit game which uses multiple Guassian Bandits and user inputs
    to simulate the pulling of levers. The bandits are shuffled initially
    so that the user is not able to determine which bandit provides the highest rewards.
    """

    def __init__(self, bandits):
        self.bandits = bandits
        np.random.shuffle(self.bandits)
        self.reset_game()

    def play(self, choice: int):
        """Simulate one round of the game."""
        reward = self.bandits[choice - 1].pull_lever()
        self.total_reward += reward
        self.n_played += 1

        return reward

    def user_play(self):
        """Allows the user to begin the game."""
        self.reset_game()
        print("Game started. " + "Enter 0 as input to end the game.")

        while True:
            print(f"\n -- Round {self.n_played}")
            choice = int(input("Choose a machine " + f"from 1 to {len(self.bandits)}:"))

            if choice in range(1, len(self.bandits)):
                reward = self.play(choice)
                print(f"Machine {choice} gave " + f"a reward of {reward}")
                avg_reward = self.total_reward / self.n_played
                print(f"Your average reward so far is {avg_reward}")

            else:
                break

        print("Game has ended.")

        if self.n_played > 0:
            print(f"Total reward is {self.total_reward} after {self.n_played} round(s)")
            avg_reward = self.total_reward / self.n_played
            print(f"Average reward is {avg_reward}")

    def reset_game(self):
        """Delete all cached values for a new game."""
        self.rewards = []
        self.total_reward = 0
        self.n_played = 0
