# SAE-fmincon

SAE-fmincon is a Python library wherein the multi-objective formula SAE system design problem is modelled as a constrained minimization problem. It is based on the problem detailed in [Design of Complex Engineered Systems Using Multi-Agent Coordination](https://asmedigitalcollection.asme.org/computingengineering/article/18/1/011003/366472/Design-of-Complex-Engineered-Systems-Using-Multi) with minor modifications.

A car is defined using a 39-dimensional vector of continuous and integer parameters. The objective to be minimized is a weighted sum of 11 sub-objectives. The minimization is subject to 3 types of inequality constraints:
1. Fixed bounds: lb ≤ x ≤ ub
2. Linear inequalities: A*x ≤ b
3. Non linear inqualities: c(x) ≤ 0

## Environment setup

<!-- -->
  
SAE-fmincon has been tested with python 3.9.1. We recommend creating a new Anaconda environment and installing the requirements.

```bash
conda create -n myEnvironment python=3.9.1
conda activate myEnvironment
conda install pip
pip install numpy==1.20.1
pip install pandas==1.2.2
pip install openpyxl==3.0.6
```

## Usage
Run in your favorite python compiler:

```python
import SAE.fmincon

# generate a random car that always satisfies constraints_bound and constraints_lin_ineq
test_car = SAE.fmincon.car()

# generate a random car that satisfies all constraints
test_car = SAE.fmincon.generate_feasible()

# get the 39-dimensional vector that defines the car
# (0-18, and 29-38 are continuous parameters while 19-28 are integer parameters)
test_car.get_vec()

# get a specific parameter
test_car.get_param(9)

# evaluate the true objectives - this function takes in 3 arguments

# first argument - an array of objective weights - use weightsNull for equal weights; use weights1, weights2 or weights3 for
# weights used in the referenced paper; define a custom numpy array of shape (11,)

# second argument - with_subobjs = True (default) or False, if False a scalar value of the weighted objective is returned,
# if True a tuple with the first element as the weighted objective and the second as an array of sub-objectives is returned

# thrid argument - tominimize_and_scaled = True (default) or False, if False the sub-objectives array holds physically
# meaningful values, if True the maximization sub-objectives are negated and all of them are scaled
# (These are the values that are involved in composing the weighted objective)

test_car.objectives(SAE.fmincon.weights1, with_subobjs=True, tominimize_and_scaled=True)

# set a parmeter value - 6th paramter as 0.5
test_car.set_param(6, 0.5)

# feed a vector to set all parameter values
dummy_vector = [1]*39
test_car.set_vec(dummy_vector)

# evaluate constraint violation penalites (square penalty)
test_car.constraints_bound()
test_car.constraints_lin_ineq()
test_car.constraints_nonlin_ineq()
```
