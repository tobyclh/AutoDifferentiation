import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from functools import lru_cache

# Initial condition

TIMESTEPS = np.arange(1, 1001) / 1e4
# print(f'Timesteps : {TIMESTEPS.min()}')
total_steps = len(TIMESTEPS)
Ts = [0.2, 0.3, 0.4]
y_0 = 2

@lru_cache(None)
def dydt(y, t):
    return (1 + y**2)/(1+t**2)

@lru_cache(None)
def real_func(t):
    return np.tan(np.arctan(2) + np.arctan(t))

@lru_cache(None)
def euler(y_n, t_n, timestep):
    return y_n + dydt(y_n, t_n) * timestep

@lru_cache(None)
def RungeKutta(y_n, t_n, timestep):
    k1 = dydt(y_n, t_n)
    k2 = dydt(y_n + timestep/2 * k1, t_n + timestep/2)
    k3 = dydt(y_n + timestep/2 * k2, t_n + timestep/2)
    k4 = dydt(y_n + timestep * k3, t_n + timestep)
    return y_n + timestep / 6 * (k1 + 2*k2 + 2*k3 + k4)

# plot the estimations


for T in Ts: #[0.2, 0.3, 0.4]
    for func in [euler, RungeKutta]:
        losses = np.zeros(total_steps) #set up array
        for i, t in enumerate(TIMESTEPS):
            y = y_0
            _t = 0
            previous_y = None
            while _t*t <= T:
                previous_y = y
                y = func(y, _t * t, t)
                _t += 1
            y = previous_y
            _t -= 1
            last_step_size = (T - _t * t)
            if last_step_size > 0:
                y = func(y, _t * t, last_step_size)
            func_val = real_func(T)
            loss = np.abs(func_val - y)
            losses[i] = loss
        plt.plot(TIMESTEPS, losses, label=f"{func.__name__}, t={T}")
plt.xlabel('Timestep (Δx)')
plt.ylabel('Error value')
plt.xscale('log')
plt.title('Error plot')
plt.legend()
plt.show()
