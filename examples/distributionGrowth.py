import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import DataManager as dm

def chi_square(true_val, test_val, std):
	return np.sum(np.square((test_val - true_val)/std))

def gaussian(x, parameters):
	a = parameters[0]
	mean = parameters[1]
	var = parameters[2]

	return a*np.exp(-np.square(x-mean)/(2*var))

frame_diff_start = 10
frame_diff_end = 200
frame_diff_interval = 10

#fit_coeffs = [] #Array of tuples with values corresponding to fit gaussians
fit_coeffs = np.zeros((0,3)) 
delays = [] #Array of delays
red_chi_squares = [] #Array of wellness of fits 

data = dm.dataset(fname = "test1_2_1_4_Camera_tr_Track")
print(f"Frame rate: {data.fps}")
print(f"Exposure: {data.exposure}")
print(f"Particles tracked in recording: {data.particles_tracked}")

for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
	#generate histogram of displacements after frame_diff frames
	data.displacement_histo(frame_diff, "x")
	zero_count_indices = []
	for i, count in enumerate(data.disp_histo_counts):
		if count == 0:
			zero_count_indices.append(i)
	data.disp_histo_counts = np.delete(data.disp_histo_counts, zero_count_indices)
	data.disp_histo_pos = np.delete(data.disp_histo_pos, zero_count_indices)
	#Produce initial fitting coefficients
	max_index = np.argmax(data.disp_histo_counts)
	max_counts = data.disp_histo_counts[max_index]
	max_pos = data.disp_histo_pos[max_index]
	fwhm_hi, fwhm_lo = 0.0, 0.0

	test_counts = max_counts
	test_index = max_index
	while test_counts > max_counts/2:
		test_index+=1
		test_counts = data.disp_histo_counts[test_index]
	fwhm_hi = data.disp_histo_pos[test_index]

	test_counts = max_counts
	test_index = max_index
	while test_counts > max_counts/2:
		test_index-=1
		test_counts = data.disp_histo_counts[test_index]
	fwhm_lo = data.disp_histo_pos[test_index]

	init_coeffs = [max_counts, max_pos, np.square((fwhm_hi - fwhm_lo)/2.35)]
	#print(init_coeffs)
	#Fit a Gaussian to the histogram and extract results
	minimisation = fmin(lambda coeffs: chi_square(data.disp_histo_counts, gaussian(data.disp_histo_pos, coeffs), np.sqrt(data.disp_histo_counts)), init_coeffs, full_output = True, disp = False)
	#fit_coeffs.append(minimisation[0])
	fit_coeffs = np.vstack((fit_coeffs, minimisation[0]))
	red_chi_squares.append(minimisation[1]/(len(data.disp_histo_counts) - 3))
	delays.append(frame_diff/data.fps)
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data.disp_histo_pos, data.disp_histo_counts, c="b")
	x = np.linspace(data.disp_histo_pos[0], data.disp_histo_pos[-1], 500)
	y = gaussian(x, minimisation[0])
	ax.plot(x, y, c="r")
	plt.show()
	'''
'''
print(len(fit_coeffs))
print(len(delays))
print(len(red_chi_squares))
'''

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.scatter(delays, fit_coeffs[:,1])
ax2.scatter(delays, red_chi_squares, c='r')

ax1.set_ylabel("Variance")
ax2.set_ylabel("Reduced chi squared")
ax2.set_xlabel("Delay in sampled frames/ms")

plt.show()