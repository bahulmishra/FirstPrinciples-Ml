import numpy as np
from model_manager import SoftSVMRegressor, HardSVMRegressor, LinearRegressionGD, _r2_score

np.random.seed(42)
X = np.random.randn(100, 3)
true_w = np.array([1.5, -2.0, 0.5])
y = X @ true_w + 3.0 + np.random.randn(100) * 0.1

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

print("Linear Regression:")
lr = LinearRegressionGD(learning_rate=0.01, epochs=1000)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print("R2:", _r2_score(y_test, pred))

print("Soft SVM (fixing objective/gradient):")
class FixedSoftSVM(SoftSVMRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        prev_loss = float("inf")

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            residuals = y - y_pred

            hinge = np.maximum(0, np.abs(residuals) - self.epsilon)
            
            # Use mean for BOTH terms or sum for BOTH terms. 
            # If we use mean for hinge, we must use 1/m for regularization too.
            # Objective: 1/(2m) ||w||^2 + C * mean(hinge) (equivalent to 1/m * standard SVM eq if C = C_real/m)
            # Standard SVM Eq: 1/2 ||w||^2 + C * sum(hinge) = m * ( 1/(2m) ||w||^2 + C * mean(hinge) )
            loss = (0.5 / m) * float(np.dot(self.w, self.w)) + self.C * float(np.mean(hinge))
            self.loss_history.append(loss)

            mask_pos = residuals < -self.epsilon
            mask_neg = residuals > self.epsilon

            grad_w = self.w.copy() / m
            grad_b = 0.0
            grad_w += self.C / m * (X[mask_pos].sum(axis=0) - X[mask_neg].sum(axis=0))
            grad_b += self.C / m * float(mask_pos.sum() - mask_neg.sum())

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
        return self

svm = FixedSoftSVM(learning_rate=0.01, epochs=1000, C=1.0)
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
print("R2:", _r2_score(y_test, pred))

print("Hard SVM:")
hsvm = HardSVMRegressor(learning_rate=0.01, epochs=1000)
hsvm.fit(X_train, y_train)
pred = hsvm.predict(X_test)
print("R2:", _r2_score(y_test, pred))

