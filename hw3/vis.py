import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

_range = 100
time_steps = 5000
x_0 = 50
x = np.arange(_range)
y = np.zeros([_range])
fig, ax = plt.subplots()
ln, = plt.plot([], [])

def func(i):
    return 0 if i >= x_0 else 1

def init():
    #initialize
    for i in range(_range):
        y[i] = func(x[i])
    ax.set_xlim(0, _range)
    ax.set_ylim(0-5, 2+5)

    return ln,

def get_phi(phi_i, phi_im1, phi_ip1):
    return phi_i + 1/4*(phi_im1-phi_ip1)

def update(_):
    for n in range(_range):
        _y = y.copy()
        _previous = _y[n-1] if n > 1 else 1
        _next = _y[n+1] if n < _range - 1 else 0
        y[n] = get_phi(_y[n], _previous, _next)
    ln.set_data(x, y)
    return ln,

ani = FuncAnimation(fig, update, frames=time_steps,
                    init_func=init, blit=True, interval=10)
plt.show()
# ani.save('anim.gif', fps=10, writer="imagemagick")
