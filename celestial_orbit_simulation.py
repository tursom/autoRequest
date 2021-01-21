import _thread
import queue

import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt

G = 6.67259e-11


def calc_point(x, y, z):
    x = np.tile(x, (len(x), 1))
    y = np.tile(y, (len(y), 1))
    z = np.tile(z, (len(z), 1))
    x -= np.transpose(x)
    y -= np.transpose(y)
    z -= np.transpose(z)
    return x, y, z


def unit(x, y, z, r=None):
    if r is None:
        r = x ** 2 + y ** 2 + z ** 2
    x, y, z = x / r, y / r, z / r
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    z[np.isnan(z)] = 0
    # x *= filter
    # y *= filter
    # z *= filter
    return x, y, z


a_fix = 100.5


def loop(mass, x, y, z, vx, vy, vz, dt, repeat=1):
    dt_base = dt ** 2
    for i in range(repeat):
        (px, py, pz) = calc_point(x, y, z)
        r2 = (px ** 2 + py ** 2 + pz ** 2)
        r = np.sqrt(r2)
        (ux, uy, uz) = unit(px, py, pz, r)
        a = G * mass / r2.transpose()
        a[np.isinf(a)] = 0
        ax, ay, az = a * ux, a * uy, a * uz
        ax, ay, az = ax.sum(axis=1), ay.sum(axis=1), az.sum(axis=1)
        # print(r + np.eye(len(r)))
        # print(a)
        dt = 1 / np.min(r + np.eye(len(r)) * np.max(r)) / np.max(a) * dt_base
        # print(dt)
        x, y, z = \
            x + vx * dt + 0.5 * ax * dt ** 2, \
            y + vy * dt + 0.5 * ay * dt ** 2, \
            z + vz * dt + 0.5 * az * dt ** 2
        vx, vy, vz = vx + ax * dt, vy + ay * dt, vz + az * dt
    return x, y, z, vx, vy, vz


planet_count = 20
1 / np.pi / 365 / 24
dt = 1000
animation_step = 300
point_base = 100_000_000
speed_base = 30
# line_frame = 10_0000
line_frame = 300

filter = 1 + -np.eye(planet_count)
massive = np.random.randn(planet_count) * np.random.random() * np.float64(100_000_000_000_000_000_000_000)
massive = np.abs(massive)
x = np.random.randn(planet_count) * np.random.random() * point_base
y = np.random.randn(planet_count) * np.random.random() * point_base
z = np.random.randn(planet_count) * np.random.random() * point_base
vx = np.random.randn(planet_count) * speed_base
vy = np.random.randn(planet_count) * speed_base
vz = np.random.randn(planet_count) * speed_base

# massive[1] = massive[0] * 1e3
# x[1], y[1], z[1], vx[1], vy[1], vz[1] = 0, 0, 0, 0, 0, 0
# x[1], y[1], z[1], vx[1], vy[1], vz[1] = -x[0], -y[0], -z[0], -vx[0], -vy[0], -vz[0]

# massive, x, y, z, vx, vy, vz = np.asarray(
#     ((0.001, 1 / G, 0.001),
#      (1, 0, 2),
#      (0, 0, 0),
#      (0, 0, 0),
#      (0, 0, 0),
#      (1, 0, 0),
#      (0, 0, 1 / np.sqrt(2))
#      ), dtype=np.float64)
print(massive, x, y, z, vx, vy, vz, sep='\n')


def avg(max, min):
    d = max - min
    return max + d, min - d


class UpdatePoints:
    def __init__(self, point_ani, massive, x, y, z, vx, vy, vz, dt,
                 times=1, lines=None, line_frame=100, ax=None, ax_frame=-1):
        self.point_ani = point_ani
        self.massive = massive
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.dt = dt
        self.times = times
        self.lines = lines
        self.line_frame = line_frame
        self.ax = ax
        self.ax_frame = ax_frame
        self.cache = queue.Queue(10)
        # self.update_cache(10)
        _thread.start_new_thread(self.update_cache, (-1,))

    def update_cache(self, step=10):
        print("update cache,step=", step)
        if step >= 0:
            for j in range(step):
                self.x, self.y, self.z, self.vx, self.vy, self.vz = loop(self.massive, self.x, self.y, self.z, self.vx,
                                                                         self.vy, self.vz, self.dt, self.times)
                self.cache.put((self.x, self.y, self.z), timeout=10)
        else:
            while True:
                self.x, self.y, self.z, self.vx, self.vy, self.vz = loop(self.massive, self.x, self.y, self.z, self.vx,
                                                                         self.vy, self.vz, self.dt, self.times)
                self.cache.put((self.x, self.y, self.z), timeout=10)

    def update_axis(self, x, y, z):
        if self.ax is not None:
            size = len(x)
            if size <= 4:
                x_max, x_min = avg(np.max(x), np.min(x))
                y_max, y_min = avg(np.max(y), np.min(y))
                z_max, z_min = avg(np.max(z), np.min(z))

                self.ax.axis((x_max, x_min, y_max, y_min))
                self.ax.set_zlim(z_max, z_min)
            else:
                x_sort, y_sort, z_sort = np.sort(x), np.sort(y), np.sort(z)
                x_sort = x_sort[size >> 2: (size >> 1) + (size >> 2)]
                y_sort = y_sort[size >> 2: (size >> 1) + (size >> 2)]
                z_sort = z_sort[size >> 2: (size >> 1) + (size >> 2)]
                self.ax.axis((np.max(x_sort), np.min(x_sort), np.max(y_sort), np.min(y_sort)))
                self.ax.set_zlim(np.max(z_sort), np.min(z_sort))

    def update_points(self, num):
        line_frame = self.line_frame
        print("frame:", num)
        # if self.cache.empty():
        #     self.update_cache(3)
        (x, y, z) = self.cache.get(timeout=1000)
        if 0 <= self.ax_frame < num:
            self.update_axis(x, y, z)

        # print(x, y, z)
        for i in range(len(self.point_ani)):
            point_ani = self.point_ani[i]
            point_ani.set_data_3d(x[i], y[i], z[i])
            # point_ani.set_data(x[i], y[i])
            # point_ani.set_3d_properties(z[i], 'z')
            if self.lines is not None:
                line = self.lines[i]
                # print(dir(line))
                lx, ly, lz = line.get_data_3d()
                lx = np.append(lx, x[i])
                ly = np.append(ly, y[i])
                lz = np.append(lz, z[i])
                lx, ly, lz = lx[-line_frame:], ly[-line_frame:], lz[-line_frame:]
                # print(line.get_data_3d())
                line.set_data_3d(lx, ly, lz)
                # line.set_3d_properties(z[i], 'z')
        return self.point_ani[0],


fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.axis((3, -3, 3, -3))
# ax.set_zlim(3, -3)

points = []
lines = []
for i in range(planet_count):
    point, = ax.plot3D(x[i], y[i], z[i], "o-")
    points.append(point)
    # print(dir(point))
    # print(point.get_color())
    lines.append(ax.plot3D([x[i]], [y[i]], [z[i]], 'b-', color=point.get_color(), lw=1)[0])
# plt.grid(ls="--")
updatePoints = UpdatePoints(points, massive, x, y, z, vx, vy, vz, dt,
                            times=animation_step, lines=lines, line_frame=line_frame, ax=ax,
                            ax_frame=-1)
# ax_frame=300)
ani = animation.FuncAnimation(fig, updatePoints.update_points, 1800)


def call_back(event):
    axtemp = event.inaxes
    if axtemp is None:
        return
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()
    z_min, z_max = axtemp.get_zlim()
    dx = (x_max - x_min) / 10
    dy = (y_max - y_min) / 10
    dz = (z_max - z_min) / 10
    print(event.button)
    if event.button == 'up':
        axtemp.set_xlim(x_min + dx, x_max - dx)
        axtemp.set_ylim(y_min + dy, y_max - dy)
        axtemp.set_zlim(z_min + dz, z_max - dz)
    elif event.button == 'down':
        axtemp.set_xlim(x_min - dx, x_max + dx)
        axtemp.set_ylim(y_min - dy, y_max + dy)
        axtemp.set_zlim(z_min - dz, z_max + dz)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('scroll_event', call_back)
# fig.canvas.mpl_connect('button_press_event', call_back)

print("show image")
plt.show()
