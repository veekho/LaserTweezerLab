import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import DataManager as dm

def chi_square(true_val, test_val, std):
	return np.sum(np.square((test_val - true_val)/std))

def polynomial(x, coeffs):
	#y = a + bx
	a = coeffs[0]
	b = coeffs[1]
	return a + b*x

frame_diff_start = 10
frame_diff_end = 200
frame_diff_interval = 10

data = dm.dataset(fname = "test1_2_1_4_Camera_tr_Track")

for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
	data.cumulative_travel(frame_diff, axis="x")
	#approximate starting parameters
	initial_coeffs = [0, data.cumulative_distance[-1]/data.cumulative_time[-1]]
	print(f"initial coeffs: {initial_coeffs}")
	errors = np.sqrt(2*np.linspace(1, len(data.cumulative_distance), len(data.cumulative_distance)))*data.d_px2um
	print(errors)
	print(data.cumulative_distance)
	minimisation = fmin(lambda coeffs: chi_square(data.cumulative_distance, polynomial(data.cumulative_time, coeffs), errors), initial_coeffs, full_output=True, disp=False)

	print(f"Reduced Chi Square = {minimisation[1]/(len(data.cumulative_distance)-2)}")
	print(f"Minimised coefficitents: {minimisation[0]}")
	x = np.linspace(data.cumulative_time[0], data.cumulative_time[-1], 500)
	y = polynomial(x, minimisation[0])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.errorbar(data.cumulative_time, data.cumulative_distance, errors, zorder=5)
	ax.plot(x, y, "r-", zorder=10)
	ax.set_xlabel("Time/s")
	ax.set_ylabel("Distance travelled over time/m")

	plt.show()
