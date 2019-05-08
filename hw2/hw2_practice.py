import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser("Practicing Euler algorithm")
parser.add_argument('--timestep', default=0.005, type=float, help='delta t for each step')
parser.add_argument('--time_end', default=1, type=float, help='delta t for each step')
parser.add_argument('--steps', default=None, type=int, help='override time end if given')

opt = parser.parse_args()

#Initial condition
T_0 = 0
if not opt.steps:
    T_end = opt.time_end
    total_steps = int((T_end - T_0) / opt.timestep) + 1
else:
    total_steps = opt.steps
y_0 = 1


def dydt(y, t):
    return -100 * y

def euler(y_n, t_n, timestep):
    return y_n + dydt(y_n, t_n) * timestep

def Ruken(y_n, t_n, timestep):
    k1 = dydt(y_n, t_n)
    k2 = dydt(y_n + timestep/2 * k1, t_n + timestep/2)
    k3 = dydt(y_n + timestep/2 * k2, t_n + timestep/2)
    k4 = dydt(y_n + timestep   * k3, t_n + timestep)
    return y_n + timestep / 6 * (k1 + k2 + k3 + k4)


ys = np.zeros(total_steps)
ys[0] = y_0
ts = np.zeros(total_steps) + T_0
for func in [euler, Ruken]:
    for i in range(0, total_steps-1):
        y_n = ys[i]
        t_n = ts[i]
        ys[i+1] = func(y_n, t_n, opt.timestep)
        ts[i+1] = t_n + opt.timestep
    plt.plot(ts, ys, label=f"{func.__name__}, dt={opt.timestep}")


plt.title('plt of func')
plt.legend()
plt.show()





