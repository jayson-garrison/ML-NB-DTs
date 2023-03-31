import numpy as np

class NaiveBayes:

    # X is numpy array, where first dimension corresponds to samples.
    # Within each sample is a dimension corresponding to features
    def fit(self, X, y):
        # Computing mean and variance
        n_samples, n_features = X.shape
        # This is to figure out unique elements in y
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # apply argmax_y summation_[between i = 0 and n](log(P(x_i|y)) + log(P(y))
        posteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # apply P(x_i | y) = (1/sqrt(2*pi*(variance_y)^2)) * e^(-((x_i-mean_y)^2)/(2*(variance_y)^2))
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x.astype(float) - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator