import unittest
import scipy.optimize
from sae import COTSCar, Car, weightsNull, weights1, generate_feasible
from numpy import clip

class TestFullProblem(unittest.TestCase):

    def test_random_generation(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = Car()
        test_car.objectives(weights=weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()
        test_car.cost()

    def test_feasible_generation(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = generate_feasible()
        test_car.objectives(weights=weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()
        test_car.cost()

    def test_pw(self):
        test_car = generate_feasible()
        test_car.parthworth_objectives()

    def test_minimize(self):
        def round_x(x):
            for i in range(19, 29):
                rounded = round(x[i])
                x[i] = rounded
            return x

        def objective(x):
            c = Car()
            c.set_vec(round_x(x))
            return c.objectives(weights=weightsNull, with_subobjs=False)

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
                {'type': 'eq', 'fun': penalty_1},
                {'type': 'eq', 'fun': penalty_2},
                {'type': 'eq', 'fun': penalty_3}
            ),
        )
        print(res)


class TestCOTSProblem(unittest.TestCase):

    def test_random_generation_cots(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = COTSCar()
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()
        test_car.cost()

    def test_feasible_generation_cots(self):
        # generate a random car that always satisfies constraints_bound and constraints_lin_ineq
        test_car = generate_feasible(cots=True)
        test_car.objectives(weights1, with_subobjs=True, tominimize_and_scaled=True)

        # evaluate constraint violation penalites (square penalty)
        test_car.constraints_bound()
        test_car.constraints_lin_ineq()
        test_car.constraints_nonlin_ineq()
        test_car.cost()

    def test_minimize_cots(self):

        def round_x(x):
            for i in range(len(x)):
                rounded = round(x[i])
                x[i] = rounded
            return clip(x, 0, [12, 12, 12, 12, 12, 6, 6, 20, 33, 4, 63, 191, 4, 4, 215, 215, 215])

        def objective(x):
            c = COTSCar()
            c.set_vec(round_x(x))
            return c.objectives(weights=weightsNull, with_subobjs=False)

        def penalty_1(x):
            c = COTSCar()
            c.set_vec(round_x(x))
            return c.constraints_bound()

        def penalty_2(x):
            c = COTSCar()
            c.set_vec(round_x(x))
            return c.constraints_nonlin_ineq()

        def penalty_3(x):
            c = COTSCar()
            c.set_vec(round_x(x))
            return c.constraints_lin_ineq()

        res = scipy.optimize.minimize(
            objective,
            generate_feasible(cots=True).vector,
            method='trust-constr',
            constraints=(
                {'type': 'eq', 'fun': penalty_1},
                {'type': 'eq', 'fun': penalty_2},
                {'type': 'eq', 'fun': penalty_3}
            ),
        )
        print(res)



