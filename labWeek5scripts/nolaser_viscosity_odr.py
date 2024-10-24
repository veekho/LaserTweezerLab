# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:05:36 2024

@author: q51174vk
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy import odr
from scipy.constants import k, pi
import DataManager as dm

dm.dir_data = "C:\\lasertweezers260924"
dm.exp_config = "C:\\lasertweezers260924\\experimentConfig.txt"

def chi_square(true_val, test_val, std):
	return np.sum(np.square((test_val - true_val)/std))

def gaussian(parameters, x):
	a = parameters[0]
	mean = parameters[1]
	var = parameters[2]

	return a*np.exp(-np.square(x-mean)/(2*var))

def linear(parameters, x): #y = a + bx
	a = parameters[0]
	b = parameters[1]
	return a + b*x

frame_diff_start = 2
frame_diff_end = 100
frame_diff_interval = 2

#fit_coeffs = [] #Array of tuples with values corresponding to fit gaussians
fit_coeffs = np.zeros((0,3))
fit_coeffs_unc = np.zeros((0,3))
delays = [] #Array of delays
red_chi_squares = [] #Array of wellness of fits 

data = dm.dataset(fname = "test1_2_1_3_Camera_tr_Track.002")
print(f"Frame rate: {data.fps}")
print(f"Exposure: {data.exposure}")
print(f"Particles tracked in recording: {data.particles_tracked}")

unc = data.d_px2um*np.sqrt(2)
radius = data.sphere_diameter/2

for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
	#generate histogram of displacements after frame_diff frames
	data.displacement_histo(frame_diff, "y")
	
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
	#minimisation = fmin(lambda coeffs: chi_square(data.disp_histo_counts, gaussian(data.disp_histo_pos, coeffs), np.sqrt(data.disp_histo_counts)), init_coeffs, full_output = True, disp = False)
	
	gauss = odr.Model(gaussian)
	mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc, sy=np.sqrt(data.disp_histo_counts))
	myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
	myoutput = myodr.run()
	"""
	myoutput.pprint()
	print(myoutput.beta)
	print("\n\n")
	#fit_coeffs.append(minimisation[0])
	"""
	fit_coeffs = np.vstack((fit_coeffs, myoutput.beta))
	fit_coeffs_unc = np.vstack((fit_coeffs_unc, myoutput.sd_beta))
	delays.append(frame_diff/data.fps)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data.disp_histo_pos, data.disp_histo_counts, c="b")
	x = np.linspace(data.disp_histo_pos[0], data.disp_histo_pos[-1], 500)
	y = gaussian(myoutput.beta, x)
	ax.plot(x, y, c="r")
	plt.show()
	
delays = np.array(delays)
init_grad = (fit_coeffs[-1,1]- fit_coeffs[0,1])/(delays[-1]-delays[0])
#minimisation = fmin(lambda coeffs: chi_square(fit_coeffs[:,1], linear(delays, coeffs), fit_coeffs_unc[:,1]), (0, 0.03/5), full_output=True, disp=False)

#output = odr.ODR(odr.RealData(delays, fit_coeffs[:,1], sy=fit_coeffs_unc[:,1]), odr.Model(linear), beta0=np.array([0, init_grad])).run()
my_model = odr.Model(linear)
my_data = odr.RealData(delays, fit_coeffs[:,1], sy=fit_coeffs_unc[:,1])
my_odr = odr.ODR(my_data, my_model, beta0=[0, init_grad])
output=my_odr.run()

print(f"Min values: a = {output.beta[0]}, b = {output.beta[1]}")

viscosity = 293*k*10**3/(3*pi*radius*output.beta[1])
d_viscosity = viscosity*output.sd_beta[1]/output.beta[1]
print(f"viscosity = {viscosity} +- {d_viscosity}")

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.errorbar(delays, fit_coeffs[:,1], fit_coeffs_unc[:,1], fmt="b.")
ax1.plot(delays, linear(output.beta, delays), "r-")

ax1.set_ylabel("Variance")
ax1.set_xlabel("Delay/ms")

plt.show()