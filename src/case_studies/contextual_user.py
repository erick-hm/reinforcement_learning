import numpy as np
import cufflinks as cf

cf.go_offline()
cf.set_config_file(world_readable=True, theme="white")


class UserGenerator:
    """
    A class to generate synthetic user contexts for an ad campaign.
    """

    def __init__(self):
        self.beta = {}
        self.beta["A"] = np.array([-4, -0.1, -3, -0.1])
        self.beta["B"] = np.array([-6, -0.1, 1, 0.1])
        self.beta["C"] = np.array([2, 0.1, 1, -0.1])
        self.beta["D"] = np.array([4, 0.1, -3, -0.2])
        self.beta["E"] = np.array([-0.1, 0, 0.5, -0.01])

        self.context = None

    def logistic(self, beta, context):
        f = np.dot(beta, context)
        p = 1 / (1 + np.exp(-f))
        return p

    def pull_lever(self, ad):
        if ad in ["A", "B", "C", "D", "E"]:
            p = self.logistic(self.beta[ad], self.context)
            reward = np.random.binomial(n=1, p=p)
            return reward

        else:
            raise Exception("Unknown ad.")

    def generate_user_with_context(self):
        # 0: international, 1: domestic
        location = np.random.binomial(n=1, p=0.6)

        # 0: desktop, 1: mobile
        device = np.random.binomial(n=1, p=0.6)

        # user age between 10 and 70 with mean age 34
        age = 10 + int(np.random.beta(2, 3) * 60)

        # define the context // add 1 for intercept
        self.context = [1, location, device, age]

        return self.context
