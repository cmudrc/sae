# import modules
from numpy import array, ones, pi, cos, sqrt, nan, nansum, mean
from numpy.typing import NDArray
from pandas import read_csv, read_excel, DataFrame
from random import uniform, randint
from typing import Union
import os

# directory path
SAEdir: str = os.path.dirname(__file__)

# read data into dataframes
params: DataFrame = read_excel(SAEdir + "/resources/params.xlsx", engine='openpyxl')
materials: DataFrame = read_csv(SAEdir + "/resources/materials.csv")
tires: DataFrame = read_csv(SAEdir + "/resources/tires.csv")
motors: DataFrame = read_csv(SAEdir + "/resources/motors.csv")
brakes: DataFrame = read_csv(SAEdir + "/resources/brakes.csv")
suspension: DataFrame = read_csv(SAEdir + "/resources/suspension.csv")
attenuators: DataFrame = read_csv(SAEdir + "/resources/attenuators.csv")
cabins: DataFrame = read_csv(SAEdir + "/resources/cabins.csv")
pressure: DataFrame = read_csv(SAEdir + "/resources/pressure.csv")
wings: DataFrame = read_csv(SAEdir + "/resources/wings.csv")

# constants
v_car: float = 26.8  # m/s
w_e: float = 3600 * 2 * pi / 60  # radians/sec
rho_air: float = 1.225  # kg/m3
r_track: float = 9.0  # m
P_brk: float = 10 ** 7  # Pascals
C_dc: float = 0.04  # drag coefficient of cabin
gravity: float = 9.81  # m/s^2
y_suspension: float = 0.05  # m
dydt_suspension: float = 0.025  # m/s

# approximate min max values of objectives
mins_to_scale = [9.54413093e+01, 1.15923589e-01, 4.82833103e+00, 6.29879814e-03, 0.00000000e+00, 1.63642506e+06,
                 4.04892567e-03, 1.90828686e-02, 5.90877604e+00, 9.81000915e+00, 3.38533564e-02]
maxs_to_scale = [5.59326140e+03, 9.99551980e-01, 6.33946522e+02, 3.21115722e+03, 4.35020445e+00, 6.72530217e+08,
                 1.57972935e-01, 1.50651984e+01, 5.53922616e+02, 1.87909887e+01, 9.85087184e+03]

# weights for composing objective from subobjectives
weightsNull = ones(11) / 11
weights1 = array([14, 1, 20, 30, 10, 1, 1, 10, 10, 2, 1]) / 100
weights2 = array([25, 1, 15, 20, 15, 1, 1, 15, 5, 1, 1]) / 100
weights3 = array([14, 1, 20, 15, 25, 1, 1, 10, 10, 2, 1]) / 100


class Car:
    # generates a car that satisfies constraints_bound and constraints_lin_ineq
    def __init__(self):

        # car vector with continuous and integer variables
        self.vector = []

        # continuous parameters with fixed bounds
        for i in range(19):
            temp = uniform(params.at[i, 'min'], params.at[i, 'max'])
            setattr(self, params.at[i, 'variable'], temp)
            self.vector.append(temp)

        # integer parameters
        # materials
        for i in range(5):
            temp = randint(0, 12)
            setattr(self, params.at[19 + i, 'variable'], materials.at[temp, 'q'])
            self.vector.append(temp)
        setattr(self, 'Eia', materials.at[temp, 'E'])

        # rear tires
        setattr(self, 'rear_tire', randint(0, 6))
        setattr(self, params.at[25, 'variable'], tires.at[self.rear_tire, 'radius'])
        setattr(self, params.at[26, 'variable'], tires.at[self.rear_tire, 'mass'])
        self.vector.append(self.rear_tire)

        # front tires
        setattr(self, 'front_tire', randint(0, 6))
        setattr(self, params.at[27, 'variable'], tires.at[self.front_tire, 'radius'])
        setattr(self, params.at[28, 'variable'], tires.at[self.front_tire, 'mass'])
        self.vector.append(self.front_tire)

        # engine
        setattr(self, 'engine', randint(0, 20))
        setattr(self, params.at[29, 'variable'], motors.at[self.engine, 'Power'])
        setattr(self, params.at[30, 'variable'], motors.at[self.engine, 'Length'])
        setattr(self, params.at[31, 'variable'], motors.at[self.engine, 'Height'])
        setattr(self, params.at[32, 'variable'], motors.at[self.engine, 'Torque'])
        setattr(self, params.at[33, 'variable'], motors.at[self.engine, 'Mass'])
        self.vector.append(self.engine)

        # brakes
        setattr(self, 'brakes', randint(0, 33))
        setattr(self, params.at[34, 'variable'], brakes.at[self.brakes, 'rbrk'])
        setattr(self, params.at[35, 'variable'], brakes.at[self.brakes, 'qbrk'])
        setattr(self, params.at[36, 'variable'], brakes.at[self.brakes, 'lbrk'])
        setattr(self, params.at[37, 'variable'], brakes.at[self.brakes, 'hbrk'])
        setattr(self, params.at[38, 'variable'], brakes.at[self.brakes, 'wbrk'])
        setattr(self, params.at[39, 'variable'], brakes.at[self.brakes, 'tbrk'])
        self.vector.append(self.brakes)

        # suspension
        setattr(self, 'suspension', randint(0, 4))
        setattr(self, params.at[40, 'variable'], suspension.at[self.suspension, 'krsp'])
        setattr(self, params.at[41, 'variable'], suspension.at[self.suspension, 'crsp'])
        setattr(self, params.at[42, 'variable'], suspension.at[self.suspension, 'mrsp'])
        setattr(self, params.at[43, 'variable'], suspension.at[self.suspension, 'kfsp'])
        setattr(self, params.at[44, 'variable'], suspension.at[self.suspension, 'cfsp'])
        setattr(self, params.at[45, 'variable'], suspension.at[self.suspension, 'mfsp'])
        self.vector.append(self.suspension)

        # continuous parameters with variable bounds
        setattr(self, 'wrw', uniform(0.3, 3 - 2 * self.rrt))
        setattr(self, 'yrw', uniform(.5 + self.hrw / 2, 1.2 - self.hrw / 2))
        setattr(self, 'yfw', uniform(0.03 + self.hfw / 2, .25 - self.hfw / 2))
        setattr(self, 'ysw', uniform(0.03 + self.hsw / 2, .250 - self.hsw / 2))
        setattr(self, 'ye', uniform(0.03 + self.he / 2, .5 - self.he / 2))
        setattr(self, 'yc', uniform(0.03 + self.hc / 2, 1.200 - self.hc / 2))
        setattr(self, 'lia', uniform(0.2, .7 - self.lfw))
        setattr(self, 'yia', uniform(0.03 + self.hia / 2, 1.200 - self.hia / 2))
        setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2))
        setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))

        for i in range(10):
            temp = getattr(self, params.at[46 + i, 'variable'])
            self.vector.append(temp)

    # objectives
    def objectives(self, weights=weightsNull, with_subobjs=True, tominimize_and_scaled=True):

        all_objectives = [
            self._mass, self._height_of_center_of_gravity,
            self._total_drag_force,
            self._total_down_force,
            self._acceleration,
            self._crash_force,
            self._impact_attenuator_volume,
            self._corner_velocity,
            self._breaking_distance,
            self._suspension_acceleration,
            self._pitch_moment
        ]

        objs = nan * ones(11)
        objs_physical_vals = nan * ones(11)

        if with_subobjs == True:

            for i in range(11):
                objs[i] = all_objectives[i]()
                objs_physical_vals[i] = objs[i]

        else:
            for i in range(11):
                if weights[i] != 0:
                    objs[i] = all_objectives[i]()

        for i in range(11):
            if weights[i] != 0:

                if i != 3 and i != 4 and i != 7:
                    objs[i] = all_objectives[i]()
                    objs[i] = (objs[i] - mins_to_scale[i]) / (maxs_to_scale[i] - mins_to_scale[i])

                else:
                    objs[i] = -all_objectives[i]()
                    objs[i] = (objs[i] - (-maxs_to_scale[i])) / (-mins_to_scale[i] - (-maxs_to_scale[i]))

        global_obj = nansum(objs * weights).sum()

        if with_subobjs:
            if tominimize_and_scaled:
                return global_obj, objs
            else:
                return global_obj, objs_physical_vals
        else:
            return global_obj

    # calculates penalty for violating constraints of the type lower bound < paramter value < upper bound
    def constraints_bound(self) -> NDArray[float]:
        pen1 = []

        for i in range(19):
            if getattr(self, params.at[i, 'variable']) < params.at[i, 'min']:
                pen1.append((getattr(self, params.at[i, 'variable']) - params.at[i, 'min']) ** 2)
            elif getattr(self, params.at[i, 'variable']) > params.at[i, 'max']:
                pen1.append((getattr(self, params.at[i, 'variable']) - params.at[i, 'max']) ** 2)
            else:
                pen1.append(0)
        return array(pen1)

    # calculates penalty for violating constraints of the type A.{parameters} < b, where A is a matrix and b is a vector
    # both bounds are checked in case lower bound > upper bound
    def constraints_lin_ineq(self) -> NDArray[float]:
        pen2 = []

        if self.wrw < 0.3:
            pen2.append((self.wrw - 0.3) ** 2)
        else:
            pen2.append(0)
        if self.wrw > 3 - 2 * self.rrt:
            pen2.append((self.wrw > 3 - 2 * self.rrt) ** 2)
        else:
            pen2.append(0)

        if self.yrw < .5 + self.hrw / 2:
            pen2.append((self.yrw - .5 - self.hrw / 2) ** 2)
        else:
            pen2.append(0)
        if self.yrw > 1.2 - self.hrw / 2:
            pen2.append((self.yrw - 1.2 + self.hrw / 2) ** 2)
        else:
            pen2.append(0)

        if self.yfw < 0.03 + self.hfw / 2:
            pen2.append((self.yfw - 0.03 - self.hfw / 2) ** 2)
        else:
            pen2.append(0)
        if self.yfw > .25 - self.hfw / 2:
            pen2.append((self.yfw - .25 + self.hfw / 2) ** 2)
        else:
            pen2.append(0)

        if self.ysw < 0.03 + self.hsw / 2:
            pen2.append((self.ysw - 0.03 - self.hsw / 2) ** 2)
        else:
            pen2.append(0)
        if self.ysw > .250 - self.hsw / 2:
            pen2.append((self.ysw - .250 + self.hsw / 2) ** 2)
        else:
            pen2.append(0)

        if self.ye < 0.03 + self.he / 2:
            pen2.append((self.ye - 0.03 - self.he / 2) ** 2)
        else:
            pen2.append(0)
        if self.ye > .5 - self.he / 2:
            pen2.append((self.ye - .5 + self.he / 2) ** 2)
        else:
            pen2.append(0)

        if self.yc < 0.03 + self.hc / 2:
            pen2.append((self.yc - 0.03 - self.hc / 2) ** 2)
        else:
            pen2.append(0)
        if self.yc > 1.200 - self.hc / 2:
            pen2.append((self.yc - 1.200 + self.hc / 2) ** 2)
        else:
            pen2.append(0)

        if self.lia < 0.2:
            pen2.append((self.lia - 0.2) ** 2)
        else:
            pen2.append(0)
        if self.lia > .7 - self.lfw:
            pen2.append((self.lia - .7 + self.lfw) ** 2)
        else:
            pen2.append(0)

        if self.yia < 0.03 + self.hia / 2:
            pen2.append((self.yia - 0.03 - self.hia / 2) ** 2)
        else:
            pen2.append(0)
        if self.yia > 1.200 - self.hia / 2:
            pen2.append((self.yia - 1.200 + self.hia / 2) ** 2)
        else:
            pen2.append(0)

        if self.yrsp < self.rrt:
            pen2.append((self.yrsp - self.rrt) ** 2)
        else:
            pen2.append(0)
        if self.yrsp > self.rrt * 2:
            pen2.append((self.yrsp - self.rrt * 2) ** 2)
        else:
            pen2.append(0)

        if self.yfsp < self.rft:
            pen2.append((self.yfsp - self.rft) ** 2)
        else:
            pen2.append(0)
        if self.yfsp > self.rft * 2:
            pen2.append((self.yfsp - self.rft * 2) ** 2)
        else:
            pen2.append(0)

        return array(pen2)

        # calculates penalty for violating constraints of the type f{parameters} < 0

    def constraints_nonlin_ineq(self) -> NDArray[float]:
        pen3 = []

        if (self._total_down_force() + self._mass() * gravity - 2 * self.__suspension_force(self.kfsp,
                                                                                            self.cfsp) - 2 * self.__suspension_force(
                self.krsp, self.crsp) < 0):
            pen3.append((self._total_down_force() + self._mass() * gravity - 2 * self.__suspension_force(self.kfsp,
                                                                                                         self.cfsp) - 2 * self.__suspension_force(
                self.krsp, self.crsp)) ** 2)
        else:
            pen3.append(0)

        if self._pitch_moment() < 0:
            pen3.append(self._pitch_moment() ** 2)
        else:
            pen3.append(0)

        return array(pen3)

    def set_param(self, i: int, val: Union[float, int]):
        self.vector[i] = val

        if i < 19:
            setattr(self, params.at[i, 'variable'], val)
        elif i < 24:
            setattr(self, params.at[i, 'variable'], materials.at[val, 'q'])
            if i == 23:
                setattr(self, 'Eia', materials.at[val, 'E'])
        elif i == 24:
            setattr(self, params.at[25, 'variable'], tires.at[val, 'radius'])
            setattr(self, params.at[26, 'variable'], tires.at[val, 'mass'])
        elif i == 25:
            setattr(self, params.at[27, 'variable'], tires.at[val, 'radius'])
            setattr(self, params.at[28, 'variable'], tires.at[val, 'mass'])
        elif i == 26:
            setattr(self, params.at[29, 'variable'], motors.at[val, 'Power'])
            setattr(self, params.at[30, 'variable'], motors.at[val, 'Length'])
            setattr(self, params.at[31, 'variable'], motors.at[val, 'Height'])
            setattr(self, params.at[32, 'variable'], motors.at[val, 'Torque'])
            setattr(self, params.at[33, 'variable'], motors.at[val, 'Mass'])
        elif i == 27:
            setattr(self, params.at[34, 'variable'], brakes.at[val, 'rbrk'])
            setattr(self, params.at[35, 'variable'], brakes.at[val, 'qbrk'])
            setattr(self, params.at[36, 'variable'], brakes.at[val, 'lbrk'])
            setattr(self, params.at[37, 'variable'], brakes.at[val, 'hbrk'])
            setattr(self, params.at[38, 'variable'], brakes.at[val, 'wbrk'])
            setattr(self, params.at[39, 'variable'], brakes.at[val, 'tbrk'])
        elif i == 28:
            setattr(self, params.at[40, 'variable'], suspension.at[val, 'krsp'])
            setattr(self, params.at[41, 'variable'], suspension.at[val, 'crsp'])
            setattr(self, params.at[42, 'variable'], suspension.at[val, 'mrsp'])
            setattr(self, params.at[43, 'variable'], suspension.at[val, 'kfsp'])
            setattr(self, params.at[44, 'variable'], suspension.at[val, 'cfsp'])
            setattr(self, params.at[45, 'variable'], suspension.at[val, 'mfsp'])
        else:
            setattr(self, params.at[i + 17, 'variable'], val)

    def set_vec(self, vec: list[Union[float, int]]):
        for i in range(39):
            self.set_param(i, vec[i])

    def get_param(self, i: int) -> Union[float, int]:
        return self.vector[i]

    def get_vec(self) -> list[Union[float, int]]:
        return self.vector

    # objective 1 - mass (minimize)
    def _mass(self) -> float:
        return (self.__rear_wing_mass()
                + self.__front_wing_mass()
                + 2 * self.__side_wing_mass()
                + 2 * self.mrt
                + 2 * self.mft
                + self.me
                + self.__mc()
                + self.__impact_attenuator_mass()
                + 4 * self.__mbrk()
                + 2 * self.mrsp
                + 2 * self.mfsp)

    # objective 2 - centre of gravity height (minimize)
    def _height_of_center_of_gravity(self) -> float:
        t1 = (self.__rear_wing_mass() * self.yrw
              + self.__front_wing_mass() * self.yfw
              + self.me * self.ye
              + self.__mc() * self.yc
              + self.__impact_attenuator_mass() * self.yia) / self._mass()
        t2 = 2 * (self.__side_wing_mass() * self.ysw
                  + self.mrt * self.rrt
                  + self.mft * self.rft
                  + self.__mbrk() * self.rft
                  + self.mrsp * self.yrsp
                  + self.mfsp * self.yfsp) / self._mass()
        return t1 + t2

    # objective 3 - total drag (minimize)
    def _total_drag_force(self) -> float:
        cabin_drag = self.__drag_force(self.wc, self.hc, rho_air, v_car, C_dc)
        rear_wing_drag = self.__wing_drag_force(self.wrw, self.hrw, self.lrw, self.arw, rho_air, v_car)
        front_wing_drag = self.__wing_drag_force(self.wfw, self.hfw, self.lfw, self.afw, rho_air, v_car)
        side_wing_drag = self.__wing_drag_force(self.wsw, self.hsw, self.lsw, self.asw, rho_air, v_car)
        return rear_wing_drag + front_wing_drag + 2 * side_wing_drag + cabin_drag

    # objective 4 - total downforce (maximize)
    def _total_down_force(self) -> float:
        down_force_rear_wing = self.__wing_down_force(self.wrw, self.hrw, self.lrw, self.arw, rho_air, v_car)
        down_force_front_wing = self.__wing_down_force(self.wfw, self.hfw, self.lfw, self.afw, rho_air, v_car)
        down_force_side_wing = self.__wing_down_force(self.wsw, self.hsw, self.lsw, self.asw, rho_air, v_car)
        return down_force_rear_wing + down_force_front_wing + 2 * down_force_side_wing

    # objective 5 - acceleration (maximize)
    def _acceleration(self) -> float:
        total_resistance = self._total_drag_force() + self.__rolling_resistance(self.Prt, v_car)

        w_wheels = v_car / self.rrt
        efficiency = 1
        torque = self.T_e

        F_wheels = torque * efficiency * w_e / (self.rrt * w_wheels)

        if F_wheels < total_resistance:
            return 0

        return (F_wheels - total_resistance) / self._mass()

    # objective 6 - crash force (minimize)
    def _crash_force(self) -> float:
        return sqrt(self._mass() * v_car ** 2 * self.wia * self.hia * self.Eia / (2 * self.lia))

    # objective 7 - impact attenuator volume (minimize)
    def _impact_attenuator_volume(self) -> float:
        return self.lia * self.wia * self.hia

    # objective 8 - corner velocity in skid pad (maximize)
    def _corner_velocity(self) -> float:
        Clat = 1.6
        total_mass = self._mass()
        forces = (self._total_down_force()
                  + total_mass * gravity
                  - 2 * self.__suspension_force(self.kfsp, self.cfsp)
                  - 2 * self.__suspension_force(self.krsp, self.crsp))
        if forces < 0:
            return 0
        return sqrt(forces * Clat * r_track / total_mass)

    # objective 9 - (minimize)
    def _breaking_distance(self) -> float:
        total_mass = self._mass()
        C = .005 + 1 / self.Prt * (.01 + .0095 * ((v_car * 3.6 / 100) ** 2))

        A_brk = self.hbrk * self.wbrk
        c_brk = .37
        Tbrk = 2 * c_brk * P_brk * A_brk * self.rbrk

        # y forces:
        F_fsp = self.__suspension_force(self.kfsp, self.cfsp)
        F_rsp = self.__suspension_force(self.krsp, self.crsp)
        Fy = total_mass * gravity + self._total_down_force() - 2 * F_rsp - 2 * F_fsp
        if Fy <= 0: Fy = 1E-10
        a_brk = Fy * C / total_mass + 4 * Tbrk / (self.rrt * total_mass)
        return v_car ** 2 / (2 * a_brk)

    # objective 10 - (minimize)
    def _suspension_acceleration(self) -> float:
        total_mass = self._mass()
        return -(2 * self.__suspension_force(self.kfsp, self.cfsp)
                 - 2 * self.__suspension_force(self.krsp, self.crsp)
                 - total_mass * gravity
                 - self._total_down_force()
                 ) / total_mass

    # objective 11 - (minimize)
    def _pitch_moment(self) -> float:
        Ffsp = self.__suspension_force(self.kfsp, self.cfsp)
        Frsp = self.__suspension_force(self.krsp, self.crsp)
        down_force_rear_wing = self.__wing_down_force(self.wrw, self.hrw, self.lrw, self.arw, rho_air, v_car)
        down_force_front_wing = self.__wing_down_force(self.wfw, self.hfw, self.lfw, self.afw, rho_air, v_car)
        down_force_side_wing = self.__wing_down_force(self.wsw, self.hsw, self.lsw, self.asw, rho_air, v_car)
        lcg = self.lc
        lf = 0.5
        return (2 * Ffsp * lf + 2 * Frsp * lf + down_force_rear_wing * (lcg - self.lrw) - down_force_front_wing * (
                    lcg - self.lfw) - 2 * down_force_side_wing * (lcg - self.lsw))

    # mass of subsystems
    def __rear_wing_mass(self) -> float:
        return self.lrw * self.wrw * self.hrw * self.qrw

    def __front_wing_mass(self) -> float:
        return self.lfw * self.wfw * self.hfw * self.qfw

    def __side_wing_mass(self) -> float:
        return self.lsw * self.wsw * self.hsw * self.qsw

    def __impact_attenuator_mass(self) -> float:
        return self.lia * self.wia * self.hia * self.qia

    def __mc(self) -> float:
        return 2 * (self.hc * self.lc * self.tc + self.hc * self.wc * self.tc + self.lc * self.hc * self.tc) * self.qc

    def __mbrk(self) -> float:
        return self.lbrk * self.wbrk * self.hbrk * self.qbrk

    # rolling resistance
    def __rolling_resistance(self, tire_pressure: float, car_velocity: float) -> float:
        C = .005 + 1 / tire_pressure * (.01 + .0095 * ((car_velocity * 3.6 / 100) ** 2))
        return C * self._mass() * gravity

    def __suspension_force(self, k: float, c) -> float:
        return k * y_suspension + c * dydt_suspension

    # aspect ratio of wing
    def __wing_aspect_ratio(self, w: float, alpha: float, l: float) -> float:
        return w * cos(alpha) / l

    # lift co-effecient
    def __lift_coefficient(self, AR: float, alpha: float) -> float:
        return 2 * pi * (AR / (AR + 2)) * alpha

    # drag co-efficient
    def __drag_coefficient(self, C_lift: float, aspect_ratio: float) -> float:
        return C_lift ** 2 / (pi * aspect_ratio)

    # wing downforce
    def __wing_down_force(self, w, h, l, alpha, rho_air, v_car):
        wingAR = self.__wing_aspect_ratio(w, alpha, l)
        C_l = self.__lift_coefficient(wingAR, alpha)
        return 0.5 * alpha * h * w * rho_air * (v_car ** 2) * C_l

    # wing drag
    def __wing_drag_force(self, w, h, l, alpha, rho_air, v_car):
        wingAR = self.__wing_aspect_ratio(w, alpha, l)
        C_l = self.__lift_coefficient(wingAR, alpha)
        C_d = self.__drag_coefficient(C_l, wingAR)
        return self.__drag_force(w, h, rho_air, v_car, C_d)

    # drag
    def __drag_force(self, width: float, height: float, rho_air: float, car_velocity: float, drag_coefficient: float) -> float:
        return 0.5 * width * height * rho_air * car_velocity ** 2 * drag_coefficient


class COTSCar:
    # generates a car that satisfies constraints_bound and constraints_lin_ineq
    def __init__(self):

        self.car = Car()

        # car vector with integer variables only
        self.vector = []

        # Fix some values at average of bounds
        for i in range(19):
            temp = mean([float(params.at[i, 'min']), float(params.at[i, 'max'])])
            self.car.set_param(i, temp)

        # first 5 entries are for materials
        for i in range(5):
            temp = randint(0, 12)
            setattr(self.car, params.at[19 + i, 'variable'], materials.at[temp, 'q'])
            self.vector.append(temp)
        setattr(self.car, 'Eia', materials.at[temp, 'E'])

        # 6th entry is for rear tires
        self.car.rear_tire = randint(0, 6)
        setattr(self.car, params.at[25, 'variable'], tires.at[self.car.rear_tire, 'radius'])
        setattr(self.car, params.at[26, 'variable'], tires.at[self.car.rear_tire, 'mass'])
        self.vector.append(self.car.rear_tire)

        # 7th entry is for front tires
        self.car.front_tire = randint(0, 6)
        setattr(self.car, params.at[27, 'variable'], tires.at[self.car.front_tire, 'radius'])
        setattr(self.car, params.at[28, 'variable'], tires.at[self.car.front_tire, 'mass'])
        self.vector.append(self.car.front_tire)

        # 8th entry is engine choice
        self.car.engine = randint(0, 20)
        setattr(self.car, params.at[29, 'variable'], motors.at[self.car.engine, 'Power'])
        setattr(self.car, params.at[30, 'variable'], motors.at[self.car.engine, 'Length'])
        setattr(self.car, params.at[31, 'variable'], motors.at[self.car.engine, 'Height'])
        setattr(self.car, params.at[32, 'variable'], motors.at[self.car.engine, 'Torque'])
        setattr(self.car, params.at[33, 'variable'], motors.at[self.car.engine, 'Mass'])
        self.vector.append(self.car.engine)

        # 9th entry is brake choice
        self.car.brakes = randint(0, 33)
        setattr(self.car, params.at[34, 'variable'], brakes.at[self.car.brakes, 'rbrk'])
        setattr(self.car, params.at[35, 'variable'], brakes.at[self.car.brakes, 'qbrk'])
        setattr(self.car, params.at[36, 'variable'], brakes.at[self.car.brakes, 'lbrk'])
        setattr(self.car, params.at[37, 'variable'], brakes.at[self.car.brakes, 'hbrk'])
        setattr(self.car, params.at[38, 'variable'], brakes.at[self.car.brakes, 'wbrk'])
        setattr(self.car, params.at[39, 'variable'], brakes.at[self.car.brakes, 'tbrk'])
        self.vector.append(self.car.brakes)

        # 10th entry is suspension choice
        self.car.suspension = randint(0, 4)
        setattr(self.car, params.at[40, 'variable'], suspension.at[self.car.suspension, 'krsp'])
        setattr(self.car, params.at[41, 'variable'], suspension.at[self.car.suspension, 'crsp'])
        setattr(self.car, params.at[42, 'variable'], suspension.at[self.car.suspension, 'mrsp'])
        setattr(self.car, params.at[43, 'variable'], suspension.at[self.car.suspension, 'kfsp'])
        setattr(self.car, params.at[44, 'variable'], suspension.at[self.car.suspension, 'cfsp'])
        setattr(self.car, params.at[45, 'variable'], suspension.at[self.car.suspension, 'mfsp'])
        self.vector.append(self.car.suspension)

        # continuous parameters with variable bounds
        setattr(self.car, 'wrw', mean([0.3, 3 - 2 * self.car.rrt]))
        setattr(self.car, 'yrw', mean([0.5 + self.car.hrw / 2, 1.2 - self.car.hrw / 2]))
        setattr(self.car, 'yfw', mean([0.03 + self.car.hfw / 2, .25 - self.car.hfw / 2]))
        setattr(self.car, 'ysw', mean([0.03 + self.car.hsw / 2, .250 - self.car.hsw / 2]))
        setattr(self.car, 'ye', mean([0.03 + self.car.he / 2, .5 - self.car.he / 2]))
        setattr(self.car, 'yc', mean([0.03 + self.car.hc / 2, 1.200 - self.car.hc / 2]))
        setattr(self.car, 'lia', mean([0.2, .7 - self.car.lfw]))
        setattr(self.car, 'yia', mean([0.03 + self.car.hia / 2, 1.200 - self.car.hia / 2]))
        setattr(self.car, 'yrsp', mean([self.car.rrt, self.car.rrt * 2]))
        setattr(self.car, 'yfsp', mean([self.car.rft, self.car.rft * 2]))

    # objectives
    def objectives(self, weights=weightsNull, with_subobjs=True, tominimize_and_scaled=True):
        return self.car.objectives(weights, with_subobjs=with_subobjs, tominimize_and_scaled=tominimize_and_scaled)

    # calculates penalty for violating constraints of the type lower bound < paramter value < upper bound
    def constraints_bound(self) -> NDArray[float]:
        return self.car.constraints_bound()

    # calculates penalty for violating constraints of the type A.{parameters} < b, where A is a matrix and b is a vector
    # both bounds are checked in case lower bound > upper bound
    def constraints_lin_ineq(self) -> NDArray[float]:
        return self.car.constraints_lin_ineq()

    def constraints_nonlin_ineq(self) -> NDArray[float]:
        return self.car.constraints_nonlin_ineq()

    def set_param(self, i: int, val: int):
        self.vector[i] = val

        if i < 5:
            setattr(self, params.at[i, 'variable'], materials.at[val, 'q'])
            if i == 4:
                setattr(self, 'Eia', materials.at[val, 'E'])
        elif i == 5:
            setattr(self, params.at[25, 'variable'], tires.at[val, 'radius'])
            setattr(self, params.at[26, 'variable'], tires.at[val, 'mass'])
        elif i == 6:
            setattr(self, params.at[27, 'variable'], tires.at[val, 'radius'])
            setattr(self, params.at[28, 'variable'], tires.at[val, 'mass'])
        elif i == 7:
            setattr(self, params.at[29, 'variable'], motors.at[val, 'Power'])
            setattr(self, params.at[30, 'variable'], motors.at[val, 'Length'])
            setattr(self, params.at[31, 'variable'], motors.at[val, 'Height'])
            setattr(self, params.at[32, 'variable'], motors.at[val, 'Torque'])
            setattr(self, params.at[33, 'variable'], motors.at[val, 'Mass'])
        elif i == 8:
            setattr(self, params.at[34, 'variable'], brakes.at[val, 'rbrk'])
            setattr(self, params.at[35, 'variable'], brakes.at[val, 'qbrk'])
            setattr(self, params.at[36, 'variable'], brakes.at[val, 'lbrk'])
            setattr(self, params.at[37, 'variable'], brakes.at[val, 'hbrk'])
            setattr(self, params.at[38, 'variable'], brakes.at[val, 'wbrk'])
            setattr(self, params.at[39, 'variable'], brakes.at[val, 'tbrk'])
        elif i == 9:
            setattr(self, params.at[40, 'variable'], suspension.at[val, 'krsp'])
            setattr(self, params.at[41, 'variable'], suspension.at[val, 'crsp'])
            setattr(self, params.at[42, 'variable'], suspension.at[val, 'mrsp'])
            setattr(self, params.at[43, 'variable'], suspension.at[val, 'kfsp'])
            setattr(self, params.at[44, 'variable'], suspension.at[val, 'cfsp'])
            setattr(self, params.at[45, 'variable'], suspension.at[val, 'mfsp'])


        # update continuous parameters with variable bounds since some bounds might have changed
        setattr(self.car, 'wrw', mean([0.3, 3 - 2 * self.car.rrt]))
        setattr(self.car, 'yrw', mean([0.5 + self.car.hrw / 2, 1.2 - self.car.hrw / 2]))
        setattr(self.car, 'yfw', mean([0.03 + self.car.hfw / 2, .25 - self.car.hfw / 2]))
        setattr(self.car, 'ysw', mean([0.03 + self.car.hsw / 2, .250 - self.car.hsw / 2]))
        setattr(self.car, 'ye', mean([0.03 + self.car.he / 2, .5 - self.car.he / 2]))
        setattr(self.car, 'yc', mean([0.03 + self.car.hc / 2, 1.200 - self.car.hc / 2]))
        setattr(self.car, 'lia', mean([0.2, .7 - self.car.lfw]))
        setattr(self.car, 'yia', mean([0.03 + self.car.hia / 2, 1.200 - self.car.hia / 2]))
        setattr(self.car, 'yrsp', mean([self.car.rrt, self.car.rrt * 2]))
        setattr(self.car, 'yfsp', mean([self.car.rft, self.car.rft * 2]))

    def set_vec(self, vec: list[int]):
        for i in range(9):
            self.car.set_param(i, vec[i])

    def get_param(self, i: int) -> int:
        return self.vector[i]

    def get_vec(self) -> list[int]:
        return self.vector


# generates cars until constraints_nonlin_ineq satisfied
def generate_feasible(cots: bool = False) -> Union[Car, COTSCar]:
    while True:
        feasible_car = COTSCar() if cots else Car()
        if sum(feasible_car.constraints_nonlin_ineq()) == 0:
            return feasible_car
