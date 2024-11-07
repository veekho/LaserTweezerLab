# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:05:36 2024

@author: q51174vk
testing "nolaser_viscosity_odr.py"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy import odr
from scipy.constants import k, pi
import DataManager as dm

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

#Arbitrary choice
frame_diff_start = 2
frame_diff_end = 300
frame_diff_interval = 2

#fit_coeffs = [] #Array of tuples with values corresponding to fit gaussians
fit_coeffs = np.zeros((0,3))
fit_coeffs_unc = np.zeros((0,3))
delays = [] #Array of delays

#works: "test1_1_1_5_Camera_tr_Track.001"
data = dm.dataset(fname = "test1_1_1_5_Camera_tr_Track")
print(f"Frame rate: {data.fps}")
print(f"Exposure: {data.exposure}")
print(f"Particles tracked in recording: {data.particles_tracked}")

unc = data.d_px2um*np.sqrt(2)
radius = data.sphere_diameter/2

for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
	#generate histogram of displacements after frame_diff frames
	data.displacement_histo(frame_diff, "x")
	
	#Remove empty bins
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
	'''
	minimisation = fmin(lambda coeffs: chi_square(data.disp_histo_counts, gaussian(data.disp_histo_pos, coeffs), np.sqrt(data.disp_histo_counts)), init_coeffs, full_output = True, disp = False)
	coeff_min, chi_min = np.array(minimisation[0]), minimisation[1]
	std_coeff_min = []
	for i in range(3):
		min_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,1], linear(coeffs, delays), fit_coeffs_unc[:,1]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[i], coeff_min, full_output=False, disp=False)[i]
		max_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,1], linear(-coeffs, delays), fit_coeffs_unc[:,1]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[i], coeff_min, full_output=False, disp=False)[i]
		std_coeff_min.append(0.5*(max_a-min_a))
	'''
	#Orthogonal distance regression approach
	gauss = odr.Model(gaussian)
	#mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc, sy=np.sqrt(data.disp_histo_counts))
	mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc) #Are we determining the error from binning random variables correctly?
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
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data.disp_histo_pos, data.disp_histo_counts, c="b")
	x = np.linspace(data.disp_histo_pos[0], data.disp_histo_pos[-1], 500)
	y = gaussian(myoutput.beta, x)
	ax.plot(x, y, c="r")
	plt.show()
	'''

#Using minimised chi squared + 1 error ellipse to minimise 
delays = np.array(delays)
init_grad = (fit_coeffs[-1,2]- fit_coeffs[0,2])/(delays[-1]-delays[0])
minimisation = fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]), (0, init_grad), full_output=True, disp=False)
coeff_min, chi_min = minimisation[0], minimisation[1]
print(f"Reduced chi squared fit to linear plot: {chi_min/(len(fit_coeffs[:,2]-2))}")
coeff_min = np.array(coeff_min)
coeff_init_upper, coeff_init_lower = coeff_min, coeff_min
offset_a = np.power(10, np.floor(np.log10(coeff_min[0])-1))
offset_b = np.power(10, np.floor(np.log10(coeff_min[1])-1))
min_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[0], coeff_min - [offset_a, 0], full_output=False, disp=False)[0]
max_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[0], coeff_min + [offset_a, 0], full_output=False, disp=False)[0]
min_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[1], coeff_min - [0, offset_b], full_output=False, disp=False)[1]
max_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[1], coeff_min + [0, offset_b], full_output=False, disp=False)[1]
std_coeff_min = [0.5*abs(max_a-min_a), 0.5*abs(max_b-min_b)]

#output = odr.ODR(odr.RealData(delays, fit_coeffs[:,1], sy=fit_coeffs_unc[:,1]), odr.Model(linear), beta0=np.array([0, init_grad])).run()
my_model = odr.Model(linear)
my_data = odr.RealData(delays, fit_coeffs[:,2], sy=fit_coeffs_unc[:,2])
my_odr = odr.ODR(my_data, my_model, beta0=[0, init_grad])
output=my_odr.run()

print(f"Min values ODR: a = {output.beta[0]} +/- {output.sd_beta[0]}, b = {output.beta[1]} +/- {output.sd_beta[1]}")
print(f"Min values X^2: a = {coeff_min[0]} +/- {std_coeff_min[0]}, b = {coeff_min[1]} +/- {std_coeff_min[1]}")
'''
viscosity = 293*k*10**3/(3*pi*radius*output.beta[1])
d_viscosity = viscosity*output.sd_beta[1]/output.beta[1]
'''
viscosity = 293*k*10**15/(3*pi*radius*coeff_min[1])
d_viscosity = viscosity*std_coeff_min[1]/coeff_min[1]

print(f"viscosity = {viscosity} +/- {d_viscosity}")

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.errorbar(delays, fit_coeffs[:,2], fit_coeffs_unc[:,2], fmt="b.")
ax1.plot(delays, linear(np.array(output.beta), delays), "r-")
ax1.plot(delays, linear(coeff_min, delays), "g-")

ax1.set_ylabel("Variance")
ax1.set_xlabel("Delay/ms")

plt.show()
