import unittest
import scipy.optimize
from sae import COTSCar, Car, weightsNull, weights1, generate_feasible


class BasicTest(unittest.TestCase):

    def test_random_generation(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = Car()
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()

    def test_feasible_generation(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = generate_feasible()
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()

    def test_random_generation_cots(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = COTSCar()
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()

    def test_feasible_generation_cots(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = generate_feasible(cots=True)
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()


class TestOptimization(unittest.TestCase):

    def test_minimize(self):

        def round_x(x):
            for i in range(19, 29):
                rounded = round(x[i])
                x[i] = rounded
            return x

        def objective(x):
            c = Car()
            c.set_vec(round_x(x))
            return c.objectives(weightsNull, with_subobjs=False)

        def penalty_1(x):
            c = Car()
            c.set_vec(round_x(x))
            return c.constraints_bound()

        def penalty_2(x):
            c = Car()
            c.set_vec(round_x(x))
            return c.constraints_nonlin_ineq()

        def penalty_3(x):
            c = Car()
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
        )
        print(res)

