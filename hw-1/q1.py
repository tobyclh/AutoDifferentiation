from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

parser = ArgumentParser()
parser.add_argument('--theta', type=float, default=0, help='angle for the flow in degree')
parser.add_argument('--time', type=int, default=1, help='how many time steps')
parser.add_argument('--delta_t', type=float, default=0.01, help='how many time steps')
parser.add_argument('--pe', type=float, default=1, help='peclet number')
parser.add_argument('--display_all', action='store_true', help='visualize all the images or just the last one')
opt = parser.parse_args()

# setup pre-defined properties
_range = 10 # the field size
_delta = 0.5 # = delta x = delta y
field_size = int(_range / _delta)
U_0 = 1
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

# Compute the flow direction in the 2 axis 
u = U_0 * np.cos(opt.theta*np.pi/180) #x
v = U_0 * np.sin(opt.theta*np.pi/180) #y

D = _delta/opt.pe

#initialization
grid = np.zeros([field_size, field_size])
grid[9:11, 9:11] = 1
if opt.display_all:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlim3d(0,1)
    surf = ax.plot_surface(X, Y, grid, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmax=1, vmin=0)
    plt.show()

def compute_T(T, Tijm1, Tijp1, Tim1j, Tip1j):
    """compute the value of T^{n+1}_{i, j}
    
    Arguments:
        T {float} -- T^{n}_{i, j}
        Tijm1 {float} -- T^{n}_{i, j-1}
        Tijp1 {float} -- T^{n}_{i, j+1}
        Tim1j {float} -- T^{n}_{i-1, j}
        Tip1j {float} -- T^{n}_{i+1, j}
    
    Returns:
        float -- T^{n+1}_{i, j}
    """

    RHS = D*(Tijm1 + Tijp1 + Tim1j + Tip1j - 4 * T)/(_delta**2)
    LHS = u*(Tip1j - Tim1j)/2/_delta + v*(Tijp1 - Tijm1)/2/_delta
    return (RHS-LHS)*opt.delta_t + T

def update():
    """update the grid inplace
    
    Returns:
        field_size x field_size array -- updated output
    """
    _grid = grid.copy()
    for i in range(field_size):
        for j in range(field_size):
            im1 = _grid[j, i-1] if i-1 >= 0 else 0
            ip1 = _grid[j, i+1] if i+1 < field_size else 0
            jm1 = _grid[j-1, i] if j-1 >= 0 else 0
            jp1 = _grid[j+1, i] if j+1 < field_size else 0
            grid[j, i] = compute_T(_grid[j, i], jm1, jp1, im1, ip1)
    return grid

n_steps = int(opt.time/opt.delta_t)
for t in range(n_steps):
    img = update()
    if opt.display_all or t == n_steps-1:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim3d(0,1)
        surf = ax.plot_surface(X, Y, img, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show()
