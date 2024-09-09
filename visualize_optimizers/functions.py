import numpy as np


class Function:
    def __init__(
        self, x_min, x_max, y_min, y_max, step_size, minima=None, initial_point=None
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step_size = step_size

        self.minima = minima
        self.initial_point = initial_point

    def __call__(self, x, y):
        raise NotImplementedError

    def grad(self, x, y):
        raise NotImplementedError


class Beale(Function):
    def __init__(self, x_min, x_max, y_min, y_max, step_size, initial_point=None):
        super().__init__(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            step_size=step_size,
            minima=np.array([[3.0, 0.5]]),
            initial_point=initial_point,
        )

    def __call__(self, x, y):
        return (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y**2) ** 2
            + (2.625 - x + x * y**3) ** 2
        )

    def grad(self, x, y):
        grad_x = -12.75+ 3 * y+ 4.5 * (y**2) + 5.25 * (y**3) + 2 * x * (3 - 2 * y - (y**2) - 2 * (y**3) + (y**4) + (y**6))

        grad_y = 6 * x * (0.5+ 1.5 * y+ 2.625 * (y**2)
                + x * (-0.333333 - 0.333333 * y - (y**2) + 0.666667 * (y**3) + (y**5))
                )

        return np.array([grad_x, grad_y])


class Booth(Function):
    def __init__(self, x_min, x_max, y_min, y_max, step_size, initial_point=None):
        super().__init__(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            step_size=step_size,
            minima=np.array([[1.0, 3.0]]),
            initial_point=initial_point,
        )

    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def grad(self, x, y):
        grad_x = 10 * x + 8 * y - 34
        grad_y = 8 * x + 10 * y - 38
        return np.array([grad_x, grad_y])


class Rosenbrock(Function):
    def __init__(self, x_min, x_max, y_min, y_max, step_size, initial_point=None):
        super().__init__(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            step_size=step_size,
            minima=np.array([[1.0, 1.0]]),
            initial_point=initial_point,
        )

    def __call__(self, x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def grad(self, x, y):
        grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
        grad_y = 20 * (y - x**2)
        return np.array([grad_x, grad_y])
