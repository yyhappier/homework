import numpy as np


class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        y_hat = 1 / (1 + np.exp(-x))
        return y_hat

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        # m : number of data points in the dataset
        # n : number of input features in the dataset
        self.m, self.n = X.shape
        self.X = X
        self.y = y
        self.lr = lr
        self.max_iter = max_iter

        # initiating
        self.w = np.zeros(self.n)
        self.b = 0
        cost_list = []
        delta = 1
        cost = 0
        i = 0
        dw = 0
        db = 0

        # implement gradient descent for optimization
        while i < self.max_iter and delta > tol:
            # y_hat formula(sigmoid function)
            z = np.dot(self.X, self.w) + self.b
            y_hat = self.sigmoid(z)

            # l1
            if self.penalty == 'l1':
                cost = -(1 / self.m) * (np.sum(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat)) \
                                        + self.gamma * np.sum(abs(self.w)))
                # derivatives
                dw = (1 / self.m) * (
                            (np.dot(self.X.T, (y_hat - self.y)) + self.w) + self.gamma * np.sum(np.sign(self.w)))
                # if false, do not update db to keep b=0
                if self.fit_intercept:
                    db = (1 / self.m) * np.sum(y_hat - self.y)
            # l2
            if self.penalty == 'l2':
                cost = -(1 / self.m) * (np.sum(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat)) \
                                        + self.gamma / 2 * np.dot(self.w.T, self.w))
                # derivatives
                dw = (1 / self.m) * (np.dot(self.X.T, (y_hat - self.y)) + self.gamma * self.w)
                # if false, do not update db to keep b=0
                if self.fit_intercept:
                    db = (1 / self.m) * np.sum(y_hat - self.y)

            # update the weights &bias
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            # if i % (max_iter / 10) == 0:
            #     print("cost after ", i, " iterations: ", cost)

            cost_list.append(cost)

            # judge tol
            if i >= 1:
                delta = abs(cost_list[i] - cost_list[i - 1])

            i += 1

        return i, cost_list

    def predict(self, X):
        """ion probabilities on a new
        collection of data points.
        Use the trained model to generate predict
        """

        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred
