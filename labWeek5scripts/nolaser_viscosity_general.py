# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:20:23 2024

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

#indices = np.zeros((0,4), dtype=int)
indices = []

with open(dm.exp_config) as file:
	for line in file:
		line = line.strip("\n").split("\t")
		try:
			if float(line[13]) == 0:
				indices.append([int(line[0]), int(line[1]), int(line[3]), int(line[6])])
				#indices = np.vstack((indices, [int(line[0]), int(line[1]), int(line[3]), int(line[6])]))
		except (ValueError, TypeError):
			pass
	
print(indices)

outfile = dm.dir_data+"\\viscosity measurements.txt"
with open(outfile, "w") as file:
	file.write("Day\tSphere\tConcentration\tRecording\tParticle num\tAxis\tViscosity\tstd(Viscosity)\n\n\n")

for day, sphere, conc, rec in indices:
	no_particles = 0
	n = no_particles
	while n <= no_particles:
		try:
			data = dm.dataset(day, sphere, conc, rec, n)
			if len(data.x_pos) < 10:
				break
			no_particles = data.particles_tracked
			
			unc = data.d_px2um*np.sqrt(2)
			radius = data.sphere_diameter/2
			for axis in ["x", "y"]:
				
				fit_coeffs = np.zeros((0,3))
				fit_coeffs_unc = np.zeros((0,3))
				delays = [] #Array of delays
				print(f"{day}\t{sphere}\t{conc}\t{rec}\t{n}\t{axis}")
				
				for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
					
					data.displacement_histo(frame_diff, axis)
					
					zero_count_indices = []
					for i, count in enumerate(data.disp_histo_counts):
						if count == 0:
							zero_count_indices.append(i)
					data.disp_histo_counts = np.delete(data.disp_histo_counts, zero_count_indices)
					data.disp_histo_pos = np.delete(data.disp_histo_pos, zero_count_indices)
					
					#Produce initial fitting coefficients
					try:
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
					except IndexError:
						break
					init_coeffs = [max_counts, max_pos, np.square((fwhm_hi - fwhm_lo)/2.35)]
					
					gauss = odr.Model(gaussian)
					mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc, sy=np.sqrt(data.disp_histo_counts))
					myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
					myoutput = myodr.run()
					
					if chi_square(data.disp_histo_counts, gaussian(myoutput.beta, data.disp_histo_pos), np.sqrt(data.disp_histo_counts))/(len(data.disp_histo_counts)-3) <3:
						fit_coeffs = np.vstack((fit_coeffs, myoutput.beta))
						fit_coeffs_unc = np.vstack((fit_coeffs_unc, myoutput.sd_beta))
						delays.append(frame_diff/data.fps)
				try:
					delays = np.array(delays)
					init_grad = (fit_coeffs[-1,1]- fit_coeffs[0,1])/(delays[-1]-delays[0])
				
					my_model = odr.Model(linear)
					my_data = odr.RealData(delays, fit_coeffs[:,1], sy=fit_coeffs_unc[:,1])
					my_odr = odr.ODR(my_data, my_model, beta0=[0, init_grad])
					output=my_odr.run()
			
					viscosity = 293*k*10**3/(3*pi*radius*output.beta[1])
					d_viscosity = viscosity*output.sd_beta[1]/output.beta[1]
				
					with open(outfile, "a") as file:
						file.write(f"{day}\t{sphere}\t{conc}\t{rec}\t{n}\t{axis}\t{viscosity}\t{d_viscosity}\n")
				except IndexError:
					pass
				
		except dm.datasetError as err:
			print(f"Indices: {day}, {sphere}, {conc}, {rec}\n{err}")
			print(type(day))
			print(type(sphere))
			print(type(conc))
			print(type(rec))
			print(type(n))
		
			
		n+=1
