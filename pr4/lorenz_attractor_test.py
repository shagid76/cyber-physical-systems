import unittest
import numpy as np

from scipy.integrate import solve_ivp
from lorenz_attractor import lorenz_deriv

xyz0 = [1.0, 1.0, 1.0]
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

sol = solve_ivp(lorenz_deriv, t_span, xyz0, t_eval=t_eval)

class TestLorenzSystem(unittest.TestCase):

    def test_solution_shape(self):
        self.assertEqual(sol.y.shape, (3, len(t_eval)), "Solution should be 3xN, where N is the length of t_eval.")

    def test_final_values(self):
        final_values = sol.y[:, -1]
        self.assertTrue(np.all(final_values > -50) and np.all(final_values < 50),
                        "Final values of the solution should be within a reasonable range.")

    def test_nonzero_solution(self):
        initial_values = sol.y[:, 0]
        final_values = sol.y[:, -1]
        self.assertFalse(np.allclose(initial_values, final_values),
                         "The solution should evolve over time.")

if __name__ == "__main__":
    unittest.main()