import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#Initial condition
# TIMESTEPS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
TIMESTEPS = [0.1, 0.01, 0.001, 0.0001]
T_0 = 0
T_end = 0.49
y_0 = 2


def dydt(y, t):
    return (1 + y**2)/(1+t**2)

def real_func(t):
    return np.tan(np.arctan(2) + np.arctan(t))

def euler(y_n, t_n, timestep):
    return y_n + dydt(y_n, t_n) * timestep

def Ruken(y_n, t_n, timestep):
    k1 = dydt(y_n, t_n)
    k2 = dydt(y_n + timestep/2 * k1, t_n + timestep/2)
    k3 = dydt(y_n + timestep/2 * k2, t_n + timestep/2)
    k4 = dydt(y_n + timestep   * k3, t_n + timestep)
    return y_n + timestep / 6 * (k1 + 2*k2 + 2*k3 + k4)

#plot the real function
total_steps = int((T_end - T_0) / TIMESTEPS[-1]) + 1
real_ys = np.zeros(total_steps)
ts = np.arange(total_steps)*TIMESTEPS[-1]
dys = np.zeros(total_steps)
for i in range(0, total_steps):
    real_ys[i] = real_func(ts[i])
    dys[i] = dydt(real_ys[i], ts[i])
plt.plot(ts, real_ys, label=f"REAL, dt={TIMESTEPS[-1]}")
# plt.plot(ts, dys, label=f"dydt, dt={TIMESTEPS[-1]}")


#plot the estimations
for timestep in TIMESTEPS:
    total_steps = int(np.ceil((T_end - T_0) / timestep)) + 1
    ts = np.arange(total_steps)*timestep

    for func in [euler, Ruken]:
        ys = np.zeros(total_steps)
        ys[0] = y_0
        for i in range(0, total_steps-1):
            y_n = ys[i]
            t_n = np.round(ts[i], 5)
            ys[i+1] = func(y_n, t_n, timestep)
            if np.isclose([0.2, 0.3, 0.4], t_n).any():
                real_yn = real_func(t_n)
                print(f'{func.__name__}\t{t_n}\t{timestep}\t{y_n}\t{real_yn}\t{np.abs(real_yn - y_n)}')
        plt.plot(ts, ys, label=f"{func.__name__}, dt={timestep}")
plt.title('plt of func')
plt.legend()
plt.show()





