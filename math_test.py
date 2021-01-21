import numpy as np
from matplotlib import pyplot as plt

# @cuda.jit
# def gpu_add(a, b, result, n):
#     idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#     if idx < n:
#         result[idx] = a[idx] + b[idx]
#
#
# def main():
#     n = 20000000
#     x = np.arange(n).astype(np.int32)
#     y = 2 * x
#
#     # 拷贝数据到设备端
#     x_device = cuda.to_device(x)
#     y_device = cuda.to_device(y)
#     # 在显卡设备上初始化一块用于存放GPU计算结果的空间
#     gpu_result = cuda.device_array(n)
#     cpu_result = np.empty(n)
#
#     threads_per_block = 1024
#     blocks_per_grid = math.ceil(n / threads_per_block)
#     start = time()
#     gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
#     cuda.synchronize()
#     print("gpu vector add time " + str(time() - start))
#     start = time()
#     cpu_result = np.add(x, y)
#     print("cpu vector add time " + str(time() - start))
#
#     if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
#         print("result correct!")
#
#
# if __name__ == "__main__":
#     main()

# def show_image(title, image, resize_time=1):
#     image = np.kron(image, np.ones((resize_time, resize_time))).astype(np.uint8)
#     cv2.imshow(title, image)
#
#
# # a = []
# # n = 16
# # resize_time = 1
# # s = 1
# # r = 1.5
# # for i in range(n):
# #     l = np.linspace((s + i * n) * r, (s + (i + 1) * n) * r - 1, n)
# #     # print(i * n, i * (n + 1) - 1, n, l)
# #     a.append(l)
# # a = np.asarray(a)
# # # print(a)
# # a *= 0.01
# # a = np.sin(a)
#
# a = cv2.imread("./test.jpg")
# # plt.subplot(131), \
# plt.imshow(a), plt.title('Original Image')
# # a = cv2.imread("./test.png", cv2.IMREAD_GRAYSCALE)
#
# # print(a)
# #
# # # show_image("a", a * 128 + 128, resize_time)
# # fft = np.fft.rfft2(a)
# # # fft = cv2.dft(np.float32(a), flags=cv2.DFT_COMPLEX_OUTPUT)[0]
# # print(fft)
# # fshift = np.fft.fftshift(fft)
# # print(fshift)
# # res = np.log(np.abs(fshift))
# # print(res)
# # plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
# plt.show()
# # fft_real = np.real(fft)
# # fft_imag = np.imag(fft)
# # fft_real_max = np.max(fft_real)
# # fft_real_min = np.min(fft_real)
# # fft_imag_max = np.max(fft_imag)
# # fft_imag_min = np.min(fft_imag)
# #
# # show_image("fft real", (fft_real + fft_real_min) / (fft_real_max - fft_real_min) * 256, resize_time)
# # show_image("fft imag", (fft_imag + fft_imag_min) / (fft_imag_max - fft_imag_min) * 256, resize_time)
# # # print(np.fft.ifft(fft))
# # cv2.waitKey(0)

# G = 6.67259e-11
#
#
# def calc_r(x, y, z):
#     x = np.tile(x, (len(x), 1))
#     y = np.tile(y, (len(y), 1))
#     z = np.tile(z, (len(z), 1))
#     x -= np.transpose(x)
#     y -= np.transpose(y)
#     z -= np.transpose(z)
#     x *= x
#     y *= y
#     z *= z
#     return np.sqrt(x + y + z)
#
#
# def calc_r2(x, y, z):
#     x = np.tile(x, (len(x), 1))
#     y = np.tile(y, (len(y), 1))
#     z = np.tile(z, (len(z), 1))
#     x -= np.transpose(x)
#     y -= np.transpose(y)
#     z -= np.transpose(z)
#     x *= x
#     y *= y
#     z *= z
#     return x + y + z
#
#
# def calc_point(x, y, z):
#     x = np.tile(x, (len(x), 1))
#     y = np.tile(y, (len(y), 1))
#     z = np.tile(z, (len(z), 1))
#     x -= np.transpose(x)
#     y -= np.transpose(y)
#     z -= np.transpose(z)
#     return x, y, z
#
#
# def unit(x, y, z, r=None):
#     if r is None:
#         r = x ** 2 + y ** 2 + z ** 2
#     return x / r, y / r, z / r
#
#
# def loop(mass, x, y, z, vx, vy, vz, dt):
#     m_tile = np.tile(mass, (len(mass), 1))
#     m_tile_transpose = np.transpose(m_tile)
#     m = m_tile * m_tile_transpose
#     (px, py, pz) = calc_point(x, y, z)
#     r2 = (px ** 2 + py ** 2 + pz ** 2)
#     (ux, uy, uz) = unit(px, py, pz, np.sqrt(r2))
#     a = G * m / r2 / m_tile_transpose
#     ax, ay, az = a * ux, a * uy, a * uz
#     ax[np.isnan(ax)] = 0
#     ay[np.isnan(ay)] = 0
#     az[np.isnan(az)] = 0
#     ax, ay, az = ax.sum(axis=1), ay.sum(axis=1), az.sum(axis=1)
#     x, y, z = x + vx * dt + 0.5 * ax * dt, y + vy * dt + 0.5 * ay * dt, z + vz * dt + 0.5 * az * dt
#     vx, vy, vz = vx + ax * dt, vy + ay * dt, vz + az * dt
#     return x, y, z, vx, vy, vz
#
#
# sample = 1
# planet_count = 100
# dt = 1
# step = 3600 * 10
# point_base = 100_000_000
# speed_base = 10
# # step = 1800
# p = int(step / 100)
# x_save = np.zeros((int(step / sample), planet_count))
# y_save = np.zeros((int(step / sample), planet_count))
# z_save = np.zeros((int(step / sample), planet_count))
# massive = np.random.randn(planet_count) * np.random.random() * np.float64(100_000_000_000_000_000_000_000)
# # massive = np.asarray([8.45001455e+22, 5.10221086e+21, 4.49085923e+22], dtype=np.float64)
# massive = np.abs(massive)
# x = np.random.randn(planet_count) * np.random.random() * point_base
# y = np.random.randn(planet_count) * np.random.random() * point_base
# z = np.random.randn(planet_count) * np.random.random() * point_base
# # x = np.asarray([-12192263.87268066, 1400939.88737432, -7530004.14530375], dtype=np.float64)
# # y = np.asarray([43783834.7261555, -13655285.51505209, -32780593.42415682], dtype=np.float64)
# # z = np.asarray([-4158673.00645479, 4595806.63983069, 3583822.32833011], dtype=np.float64)
# # y = np.zeros(planet_count)
# # z = np.zeros(planet_count)
# # vx, vy, vz = np.zeros(planet_count), np.zeros(planet_count), np.zeros(planet_count)
# vx = np.random.randn(planet_count) * speed_base
# vy = np.random.randn(planet_count) * speed_base
# vz = np.random.randn(planet_count) * speed_base
# # vx = np.asarray([-500.59145229, -843.61469736, 705.87837704], dtype=np.float64)
# # vy = np.asarray([-167.41023807, -2282.15944011, 589.70845163], dtype=np.float64)
# # vz = np.asarray([-565.11817403, 159.33682641, 456.30235333], dtype=np.float64)
#
# print("massive:", massive)
# print("x:", x, "\ny:", y, "\nz:", z)
# print("vx:", vx, "\nvy:", vy, "\nvz:", vz)
# p_state = p
# sample_state = sample
# sample_i = 0
# for i in range(step):
#     # print(f"loop in {i} seconds")
#     x, y, z, vx, vy, vz = loop(massive, x, y, z, vx, vy, vz, dt)
#     sample_state -= 1
#     p_state -= 1
#     if p_state == 0:
#         print(time.asctime(time.localtime(time.time())), int(i / step * 100), "%")
#         p_state = p
#     if sample_state == 0:
#         sample_state = sample
#         x_save[sample_i] = np.copy(x)
#         y_save[sample_i] = np.copy(y)
#         z_save[sample_i] = np.copy(z)
#         sample_i += 1
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# x_save = x_save.transpose()
# y_save = y_save.transpose()
# z_save = z_save.transpose()
# print(x_save)
# # for i in range(planet_count):
# #     ax1.plot3D(x_save[i], y_save[i], z_save[i])  # 绘制空间曲线
#
# # plt.show()
#
# points = []
#
#
# class UpdatePoints:
#     def __init__(self, point_ani, x, y, z, times=1):
#         self.point_ani = point_ani
#         self.x = x
#         self.y = y
#         self.z = z
#         self.times = times
#
#     def update_points(self, num):
#         num *= self.times
#         for i in range(len(self.point_ani)):
#             point_ani = self.point_ani[i]
#             x = self.x[i]
#             y = self.y[i]
#             z = self.z[i]
#             # if num % 5 == 0:
#             #     point_ani.set_marker("*")
#             #     point_ani.set_markersize(12)
#             # else:
#             #     point_ani.set_marker("o")
#             #     point_ani.set_markersize(8)
#
#             # print(x[num], y[num], z[num])
#             point_ani.set_data(x[num], y[num])
#             point_ani.set_3d_properties(z[num], 'z')
#         # text_pt.set_text("x=%.3f, y=%.3f" % (x[num], y[num]))
#         return self.point_ani[0],
#
#
# # x = np.linspace(0, 2 * np.pi, 100)
# # y = np.sin(x)
# # fig = plt.figure(tight_layout=True)
# # plt.plot(x, y)
#
# for i in range(len(x_save)):
#     point_ani, = ax1.plot3D(x_save[i][0], y_save[i][0], z_save[i][0], "o")
#     points.append(point_ani)
# # point_ani, = plt.plot(x[0], y[0], "o")
# plt.grid(ls="--")
# # text_pt = plt.text(4, 0.8, '', fontsize=16)
#
# # for i in range(len(x_save)):
# #     point_ani = points[i]
# animation_step = 50
# print(int(len(x_save[0]) / animation_step))
# updatePoints = UpdatePoints(points, x_save, y_save, z_save, animation_step)
# ani = animation.FuncAnimation(fig, updatePoints.update_points, int(len(x_save[0]) / animation_step))

# ani.save('sin_test3.gif', writer='imagemagick', fps=10)

x = np.linspace(0, np.pi, 10000)
y = np.tan(x)

p = plt.plot(x, y)[0]
# print(help(p.axes))
plt.axis((x[0], x[-1], 10, -10))
plt.show()
