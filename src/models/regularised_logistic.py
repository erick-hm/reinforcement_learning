import numpy as np
from scipy.optimize import minimize


class RegularisedLogisticRegression:
    def __init__(self, name: str, alpha: float, rlambda: float, n_dim: int):
        """
        Params
        ------
        name: str,
            The name of the model - a separate model will be required for each action.

        alpha: float,
            The strength of the exploration vs exploitation

        rlambda: float,
            The lambda regularisation on the parameters

        n_dim: int,
            The size of the context (number of variables)
        """
        self.name = name
        self.alpha = alpha
        self.rlamba = rlambda
        self.n_dim = n_dim

        # correspond to the beta in the bandit model
        self.w = self.get_sampled_weights()

        # the mean estimate for the weights w
        self.m = np.zeros(n_dim)

        # the inverse variance in the weights
        self.q = np.ones(n_dim) * rlambda

    def get_sampled_weights(self):
        """Generate weights assuming a Gaussian likelihood."""
        w = np.random.normal(self.m, self.alpha * self.q ** (-1 / 2))
        return w

    def loss(self, w: np.ndarray, *args):
        """
        Calculate the loss given a weight vector and a model input and target.
        The regulariser:

        regulariser = (1/2) (1/sigma^2) * (w - m)^2

        Penalises the sampled weights for being too far from the mean of the weight
        distribution.
        The prediction loss:

        loss[j] = ln(1 + exp(w . X[j])) - y[j] * (w . X[j])

        is a rewriting of the binary cross entropy loss:

        BCE = - y_j log y'_j - (1-y_j) log (1 - y'_j)

        Where y' is the estimated target, and y is the actual target. If we insert the
        logistic form for y'_j = 1 / (1 + exp(-w.X[j]))

        """
        X, y = args
        n = len(y)
        regularizer = 0.5 * np.dot(self.q, (w - self.m) ** 2)
        pred_loss = sum(
            [
                np.log(1 + np.exp(np.dot(w, X[j]))) - (y[j] * np.dot(w, X[j]))
                for j in range(n)
            ]
        )

        return regularizer + pred_loss

    def fit(self, X, y):
        """
        Fit the model on the feature data X and target y.
        """
        if y:
            X = np.array(X)
            y = np.array(y)

            minimisation = minimize(
                self.loss,
                self.w,
                args=(X, y),
                method="L-BFGS-B",
                bounds=[(-10, 10)] * 3 + [(-1, 1)],
                options={"maxiter": 50},
            )

            self.w = minimisation.x
            self.m = self.w
            p = (1 + np.exp(-np.matmul(self.w, X.T))) ** (-1)
            self.q = self.q + np.matmul(p + (1 - p), X**2)

    def calc_sigmoid(self, w, context):
        return 1 / (1 + np.exp(-np.dot(w, context)))

    def get_ucb(self, context):
        pred = self.calc_sigmoid(self.m, context)
        confidence = self.alpha * np.sqrt(np.sum((np.array(context) ** 2) / self.q))
        ucb = pred + confidence
        return ucb

    def get_prediction(self, context):
        return self.calc_sigmoid(context)

    def sample_prediction(self, context):
        w = self.get_sampled_weights()
        return self.calc_sigmoid(w, context)
