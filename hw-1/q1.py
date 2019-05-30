from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

parser = ArgumentParser()
parser.add_argument('--theta', type=float, default=0, help='angle for the flow in degree')
parser.add_argument('--time', type=int, default=0, help='compute until t=#, 0 means until it converges')
parser.add_argument('--delta_t', type=float, default=0.01, help='how many time steps')
parser.add_argument('--pe', type=float, default=1, help='peclet number')
parser.add_argument('--display_all', action='store_true', help='visualize all the images or just the last one')
opt = parser.parse_args()

# setup pre-defined properties
U_0 = 1
delta_d = 10/20 #displacement delta = delta_x = delta_y
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)
field_size = y.shape[0]

# Compute the flow direction in the 2 axis 
u = U_0 * np.cos(opt.theta*np.pi/180) #x
v = U_0 * np.sin(opt.theta*np.pi/180) #y

D = delta_d/opt.pe

#initialization
grid = np.zeros([field_size, field_size])
grid[9:11, 9:11] = 1

def draw(img):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlim3d(0,1)
    surf = ax.plot_surface(X, Y, img, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmax=1, vmin=0)
    plt.show()

if opt.display_all:
    draw(img)

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

    RHS = D*(Tijm1 + Tijp1 + Tim1j + Tip1j - 4 * T)/(delta_d**2)
    LHS = u*(Tip1j - Tim1j)/2/delta_d + v*(Tijp1 - Tijm1)/2/delta_d
    return (RHS-LHS)*opt.delta_t + T

def update():
    """compute the grid for 1 time step
    
    Returns:
        field_size x field_size array -- updated output
    """
    _grid = grid.copy()
    for i in range(field_size):
        for j in range(field_size):
            im1 = grid[j, i-1] if i-1 >= 0 else 0
            ip1 = grid[j, i+1] if i+1 < field_size else 0
            jm1 = grid[j-1, i] if j-1 >= 0 else 0
            jp1 = grid[j+1, i] if j+1 < field_size else 0
            _grid[j, i] = compute_T(grid[j, i], jm1, jp1, im1, ip1)
    return _grid

i = 0
if opt.time != 0:
    #compute a fixed number of steps
    n_steps = int(opt.time/opt.delta_t)
    def criteria(_i, img):
        return i>=n_steps
else:
    #compute until it converges
    past_error = 1e10
    def criteria(_i, img):
        error = np.abs(img-grid).mean()/np.abs(grid).mean()
        improved_rate = abs((error - past_error)/past_error)
        past_error = error
        return improved_rate < 1e-4
done = False
while not done:
    img = update()
    if opt.display_all:
        draw(img)
    i += 1
    result = criteria(i, img)
    result
    print(i, np.abs(img-grid).mean()/np.abs(grid).mean())
    grid = img


draw(grid)