#New file format: {Filename}\t{Axis}\t{Sphere radius}\t{Sphere solution volume}\t{Water volume}\t{Frame rate}\t{Exposure time}\t{Avg Gaussian X^2 red}\t{Viscosity}\t{d Viscosity}\t{mean b}\t{std b}\t{mean a}\t{std a}\t{Varience plot X^2 red}\t{Jerk X^2 red}\t{Drift X^2 red}\t{Drift grad}\t{Drift d_grad}\n

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

T = 293
d_T = 5
max_delay = 2 #seconds
min_frame_diff = 2
frame_diff_interval = 2

dm.dir_data = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\lt121124"

outfile = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\LaserTweezerLab\\nolaserResults.txt"

n_test_plots = 5

for infile in os.listdir(dm.dir_data):
	if os.stat(dm.dir_data+"\\"+infile).st_size > 1000:
		data = dm.dataset(fname = infile)
		test_plot_diffs = np.linspace(min_frame_diff, int(int(max_delay*data.fps)+1), n_test_plots, dtype=int)

		for axis in ['x', 'y']:
			fit_coeffs = np.zeros((0,3))
			fit_coeffs_unc = np.zeros((0,3))
			delays = np.zeros((0,1))
			fit_red_chi2 = np.zeros((0,1))

			for frame_diff in range(min_frame_diff, int(max_delay*data.fps)+1, frame_diff_interval):
				#Generate histogram
				data.displacement_histo(frame_diff, axis)

				#Produce initial fitting coefficients
				max_index = np.argmax(data.disp_histo_counts)
				max_counts = data.disp_histo_counts[max_index]
				max_pos = data.disp_histo_pos[max_index]
				fwhm_hi, fwhm_lo = 0.0, 0.0

				test_counts = max_counts
				upper_test_index = max_index
				while test_counts > max_counts/2:
					upper_test_index+=1
					test_counts = data.disp_histo_counts[upper_test_index]
				fwhm_hi = data.disp_histo_pos[upper_test_index]

				test_counts = max_counts
				lower_test_index = max_index
				while test_counts > max_counts/2:
					lower_test_index-=1
					test_counts = data.disp_histo_counts[lower_test_index]
				fwhm_lo = data.disp_histo_pos[lower_test_index]

				init_coeffs = [max_counts, max_pos, np.square((fwhm_hi - fwhm_lo)/2.35)]

				#Remove empty bins
				zero_count_indices = []
				for i, count in enumerate(data.disp_histo_counts):
					if count == 0:
						zero_count_indices.append(i)
				test_histo_counts = np.delete(data.disp_histo_counts, zero_count_indices)
				test_histo_pos = np.delete(data.disp_histo_pos, zero_count_indices)

				#Orthogonal distance regression
				gauss = odr.Model(gaussian)
				mydata = odr.RealData(test_histo_pos, test_histo_counts, sx=data.d_px2um*np.sqrt(2), sy=np.sqrt(test_histo_counts))
				myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
				myoutput = myodr.run()

				fit_coeffs = np.vstack((fit_coeffs, myoutput.beta))
				fit_coeffs_unc = np.vstack((fit_coeffs_unc, myoutput.sd_beta))
				delays = np.append(delays, frame_diff/data.fps)
				fit_red_chi2 = np.append(fit_red_chi2, chi_square(test_histo_counts, gaussian(myoutput.beta, test_histo_pos), np.sqrt(test_histo_counts))/(len(test_histo_pos)-3))

				if frame_diff in test_plot_diffs:
					fig = plt.figure()
					ax = fig.add_subplot(111)
					ax.errorbar(data.disp_histo_pos, data.disp_histo_counts, np.sqrt(data.disp_histo_counts), data.d_px2um*np.sqrt(2), fmt="b.")
					x_ax = np.linspace(data.disp_histo_pos[0], data.disp_histo_pos[-1], 200)
					ax.plot(x_ax, gaussian(myoutput.beta, x_ax), c='r')
					ax.set_xlabel("Displacement/um")
					ax.set_ylabel("Counts")
					ax.set_title(f"{axis} axis, {infile}")
					plt.show()

			figs = {}
			var_fig = plt.figure()
			#Variance analysis/viscosity calc
			init_grad = (fit_coeffs[-1,2]- fit_coeffs[0,2])/(delays[-1]-delays[0])
			minimisation = fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]), (0, init_grad), full_output=True, disp=False)
			coeff_min, chi_min = minimisation[0], minimisation[1]
			print(f"Reduced chi squared fit to linear plot: {chi_min/(len(fit_coeffs[:,2]-2))}")
			coeff_min = np.array(coeff_min)
			coeff_init_upper, coeff_init_lower = coeff_min, coeff_min
			offset_a = np.power(10, np.floor(np.log10(coeff_min[0])-1))
			offset_b = np.power(10, np.floor(np.log10(coeff_min[1])-1))
			min_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1), coeff_min_test, full_output=False, disp=False)[0], coeff_min - [offset_a, 0], full_output=False, disp=False)[0]
			max_a = fmin(lambda coeff_min_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1), coeff_min_test, full_output=False, disp=False)[0], coeff_min + [offset_a, 0], full_output=False, disp=False)[0]
			min_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1), coeff_min_test, full_output=False, disp=False)[1], coeff_min - [0, offset_b], full_output=False, disp=False)[1]
			max_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1), coeff_min_test, full_output=False, disp=False)[1], coeff_min + [0, offset_b], full_output=False, disp=False)[1]
			std_coeff_min = [0.5*abs(max_a-min_a), 0.5*abs(max_b-min_b)]

			var_ax = var_fig.add_subplot(111)
			var_ax.errorbar(delays, fit_coeffs[:,2], fit_coeffs_unc[:,2], fmt="r.")
			var_ax.plot(delays, linear(coeff_min, delays), "b-")
			var_ax.set_title("Variance analysis")
			var_ax.set_xlabel("Time delay/s")
			var_ax.set_ylabel("Variance")

			figs[axis] = plt.figure()
			figs[axis].suptitle(f"{axis} axis, {infile}")
	
			ax1 = figs[axis].add_subplot(311)
			ax1.scatter(delays, fit_red_chi2, c="r", marker='.')
			ax1.set_xlabel("Time delay/s")
			ax1.set_ylabel("Red X^2")

			viscosity = T*k*10**18/(1.5*pi*data.sphere_diameter*coeff_min[1])
			d_viscosity = viscosity*np.sqrt(np.square(std_coeff_min[1]/coeff_min[1]) + np.square(d_T/T))

			#Drift analysis
			init_grad = (fit_coeffs[-1,1]- fit_coeffs[0,1])/(delays[-1]-delays[0])
			minimisation = fmin(lambda coeffs: chi_square(fit_coeffs[:,1], linear(coeffs, delays), fit_coeffs_unc[:,1]), (0, init_grad), full_output=True, disp=False)
			drift_coeff, drift_chi = np.array(minimisation[0]), minimisation[1]
			offset_d = np.power(10, np.floor(np.log10(drift_coeff[1])-1))
			min_d = fmin(lambda drift_coeff_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,1], linear(coeffs, delays))), drift_coeff_test, full_output=False, disp=False)[1], drift_coeff - [0, offset_d], full_output=False, disp=False)[1]
			max_d = fmin(lambda drift_coeff_test: fmin(lambda coeffs: abs(chi_square(fit_coeffs[:,1], linear(coeffs, delays))), drift_coeff_test, full_output=False, disp=False)[1], drift_coeff + [0, offset_d], full_output=False, disp=False)[1]
			#fig2 = plt.figure()
			ax2 = figs[axis].add_subplot(312)
			ax2.errorbar(delays, fit_coeffs[:,1], fit_coeffs_unc[:,1], fmt="b.")
			ax2.plot(delays, linear(drift_coeff, delays), "r-")
			ax2.set_xlabel("Delay time/s")
			ax2.set_ylabel("Mean position/um")
			ax2.set_title("Drift analysis")
			#plt.show(block=block)

			#Jerk analysis
			data.cumulative_travel(int(max_delay*data.fps), axis)
			initial_coeffs = [0, data.cumulative_distance[-1]/data.cumulative_time[-1]]
			errors = np.sqrt(2*np.linspace(1, len(data.cumulative_distance), len(data.cumulative_distance)))*data.d_px2um
			minimisation = fmin(lambda coeffs: chi_square(data.cumulative_distance, linear(coeffs, data.cumulative_time), errors), initial_coeffs, full_output=True, disp=False)
			jerk_coeff, jerk_chi = minimisation[0], minimisation[1]
			print(f"{axis} axis, reduced Chi Square = {jerk_chi/(len(data.cumulative_time)-2)}")
	
			#fig2 = plt.figure()
			ax2 = figs[axis].add_subplot(313)
			ax2.errorbar(data.cumulative_time, data.cumulative_distance, errors, fmt="b.")
			ax2.plot(data.cumulative_time, linear(jerk_coeff, data.cumulative_time), "r-", label=f"Reduced Chi Square = {jerk_chi/(len(data.cumulative_time)-2)}")
			ax2.set_title("Cumulative travel")
			ax2.set_ylabel("Distance/um")
			ax2.set_xlabel("Time")

			print(f"{axis} axis of {infile}:\nVariance = {coeff_min[1]} +/- {std_coeff_min[1]}, Reduced Chi Squared fit: {chi_min/(len(fit_coeffs)-2)}\nViscosity = {viscosity} +/- {d_viscosity}\nGoodness of linear fits:\n Jerk: {minimisation[1]/(len(data.cumulative_distance)-2)}\tDrift: {drift_chi/(len(fit_coeffs)-2)}")
			with open(outfile, "a") as file:
				file.write(f"{infile}\t{axis}\t{data.sphere_diameter/2}\t{data.sphere_volume}\t{data.water_volume}\t{data.fps}\t{data.exposure}\t{np.mean(fit_red_chi2)}\t{viscosity}\t{d_viscosity}\t{coeff_min[1]}\t{std_coeff_min[1]}\t{coeff_min[0]}\t{std_coeff_min[0]}\t{chi_min/(len(fit_coeffs)-2)}\t{jerk_chi/(len(data.cumulative_distance)-2)}\t{drift_chi/(len(fit_coeffs)-2)}\t{drift_coeff[1]}\t{0.5*(max_d - min_d)}\n")

			figs[axis].tight_layout()

		plt.show()