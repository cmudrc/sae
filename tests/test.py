import unittest
import scipy.optimize
from SAE.fmincon import car, weightsNull, generate_feasible


class TestOptimization(unittest.TestCase):

    def test_fmincon(self):

        def round_x(x):
            for i in range(19, 29):
                rounded = round(x[i])
                x[i] = rounded
            return x

        def objective(x):
            c = car()
            c.set_vec(round_x(x))
            return c.objectives(weightsNull, with_subobjs=False)

        def penalty_1(x):
            c = car()
            c.set_vec(round_x(x))
            return c.constraints_bound()

        def penalty_2(x):
            c = car()
            c.set_vec(round_x(x))
            return c.constraints_nonlin_ineq()

        def penalty_3(x):
            c = car()
            c.set_vec(round_x(x))
            return c.constraints_lin_ineq()

        res = scipy.optimize.minimize(
            objective,
            generate_feasible().vector,
            method='trust-constr',
            constraints=(
                {'type': 'ineq', 'fun': penalty_1},
                {'type': 'ineq', 'fun': penalty_2},
                {'type': 'ineq', 'fun': penalty_3}
             ),
            callback=callbackF
        )
        print(res)