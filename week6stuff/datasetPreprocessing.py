"""
Three types of preprocessing:
	Remove small data files - bad fitting to histograms
	Jerked recordings - bad cumulative travel fitting
	Drifted recordings - (good) linear (fitting) growth in anomolous fitted Gaussian mean
	
For each file/index permutation tabulate fitting scores (reduced chi squared) - to gauge amount of erroneous data to throw away
"""
import os
import numpy as np
from scipy.optimize import fmin
from scipy import odr
import DataManager as dm
import warnings

def linear(coeffs, x):
	a, b = coeffs[0], coeffs[1]
	return a + b*x

def gaussian(coeffs, x):
	A, mean, var = coeffs[0], coeffs[1], coeffs[2]
	return A*np.exp(-np.square(x-mean)/(2*var))

def chi_square(true_val, test_val, std):
	return np.sum(np.square((test_val - true_val)/std))

def fit_centre(data):
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

	gauss = odr.Model(gaussian)
	#mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc, sy=np.sqrt(data.disp_histo_counts))
	mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=np.sqrt(2)*data.d_px2um, sy=np.sqrt(data.disp_histo_counts)) #Are we determining the error from binning random variables correctly?
	myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
	myoutput = myodr.run()

	return myoutput.beta[1], myoutput.sd_beta[1]

ct_fits_x = [] #Reduced chi squared value values fitting linear plot to cumulative travel plots (minimum delay) - X axis
ct_fits_y = [] #Reduced chi squared value values fitting linear plot to cumulative travel plots (minimum delay) - Y axis
ct_fits_xy = [] #Reduced chi squared value values fitting linear plot to cumulative travel plots (minimum delay) - both axes
mu_fits_x = [] #Reduced chi squared value values fitting linear plot to mean of gaussian fitted histogram vs delay time (frame diff = range(2,100,2)) - X axis
mu_fits_y = [] #Reduced chi squared value values fitting linear plot to mean of gaussian fitted histogram vs delay time (frame diff = range(2,100,2)) - Y axis
file_names = [] #Corresponding file names
bad_files = {}

file_num = 0
for file in os.listdir(dm.dir_data):
	data = dm.dataset(fname=file)
	file_names.append(file)
	print(f"Analysing file: {file}")
	#Jerk analysis
	try:
		frame_interval = 5
		data.cumulative_travel(frame_interval, "x")
		initial_coeffs = [0, data.cumulative_distance[-1]/data.cumulative_time[-1]]
		errors = np.sqrt(2*np.linspace(1, len(data.cumulative_distance), len(data.cumulative_distance)))*data.d_px2um
		minimisation = fmin(lambda coeffs: chi_square(data.cumulative_distance, linear(coeffs, data.cumulative_time), errors), initial_coeffs, full_output=True, disp=False)
		ct_fits_x.append(minimisation[1]/(len(data.cumulative_distance) - 2))

		data.cumulative_travel(frame_interval, "y")
		initial_coeffs = [0, data.cumulative_distance[-1]/data.cumulative_time[-1]]
		errors = np.sqrt(2*np.linspace(1, len(data.cumulative_distance), len(data.cumulative_distance)))*data.d_px2um
		minimisation = fmin(lambda coeffs: chi_square(data.cumulative_distance, linear(coeffs, data.cumulative_time), errors), initial_coeffs, full_output=True, disp=False)
		ct_fits_y.append(minimisation[1]/(len(data.cumulative_distance) - 2))

		data.cumulative_travel(frame_interval, "xy")
		initial_coeffs = [0, data.cumulative_distance[-1]/data.cumulative_time[-1]]
		errors = np.sqrt(4*np.linspace(1, len(data.cumulative_distance), len(data.cumulative_distance)))*data.d_px2um
		minimisation = fmin(lambda coeffs: chi_square(data.cumulative_distance, linear(coeffs, data.cumulative_time), errors), initial_coeffs, full_output=True, disp=False)
		ct_fits_xy.append(minimisation[1]/(len(data.cumulative_distance) - 2))

		#Drift analysis x
		centres = []
		d_centres = []
		time_diff = []
		for frame_diff in range(2, 100, 2):
			data.displacement_histo(frame_diff, "x")
			centre, d_centre = fit_centre(data)
			centres.append(centre)
			d_centres.append(d_centre)
			time_diff.append(frame_diff)

		centres = np.array(centres)
		d_centres = np.array(d_centres)
		time_diff = np.array(time_diff)/data.fps

		initial_coeffs = [0, (centres[-1]-centres[0])/(time_diff[-1]-time_diff[0])]
		minimisation = fmin(lambda coeffs: chi_square(centres, linear(coeffs, time_diff), d_centres), initial_coeffs, full_output=True, disp=False)
		mu_fits_x.append(minimisation[1]/(len(time_diff)-2))

		#Drift analysis y
		centres = []
		d_centres = []
		time_diff = []
		for frame_diff in range(2, 100, 2):
			data.displacement_histo(frame_diff, "y")
			centre, d_centre = fit_centre(data)
			centres.append(centre)
			d_centres.append(d_centre)
			time_diff.append(frame_diff)

		centres = np.array(centres)
		d_centres = np.array(d_centres)
		time_diff = np.array(time_diff)/data.fps

		initial_coeffs = [0, (centres[-1]-centres[0])/(time_diff[-1]-time_diff[0])]
		minimisation = fmin(lambda coeffs: chi_square(centres, linear(coeffs, time_diff), d_centres), initial_coeffs, full_output=True, disp=False)
		mu_fits_y.append(minimisation[1]/(len(time_diff)-2))
		file_num += 1
	except Exception as err:
		bad_files[data.file] = err
		ct_fits_x = ct_fits_x[:file_num]
		ct_fits_y = ct_fits_y[:file_num] #Reduced chi squared value values fitting linear plot to cumulative travel plots (minimum delay) - Y axis
		ct_fits_xy = ct_fits_xy[:file_num] #Reduced chi squared value values fitting linear plot to cumulative travel plots (minimum delay) - both axes
		mu_fits_x = mu_fits_x[:file_num] #Reduced chi squared value values fitting linear plot to mean of gaussian fitted histogram vs delay time (frame diff = range(2,100,2)) - X axis
		mu_fits_y = mu_fits_y[:file_num] #Reduced chi squared value values fitting linear plot to mean of gaussian fitted histogram vs delay time (frame diff = range(2,100,2)) - Y axis
		file_names = file_names[:file_num] 

for i in range(len(mu_fits_x)):
	print(f"\nFilename: {file_names[i]}\nReduced Chi squared:\nCumulative x: {ct_fits_x[i]}, y:{ct_fits_y[i]}, x&y: {ct_fits_xy[i]}\nMean position x: {mu_fits_x[i]}, y: {mu_fits_y[i]}")

