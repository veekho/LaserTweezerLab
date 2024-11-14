#New file format: 
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

def midbin(bin_edges):
	midpoints = np.zeros((0,1))
	last_edge = bin_edges[0]
	for next_edge in bin_edges[1:]:
		midpoints = np.append(midpoints, (next_edge + last_edge)/2)
		last_edge = next_edge
	return midpoints

def fit_gaussian(data_in, axis):
	#Generate histogram
	buffer = []
	buffer_means = np.zeros(1)
	pos_array = np.zeros((0,1))

	if axis == 'x' or axis == 'X':
		pos_array = data_in.x_pos*data_in.px2um
	elif axis == 'y' or axis == 'Y':
		pos_array = data_in.y_pos*data_in.px2um

	histo, bin_edges = np.histogram(pos_array - np.mean(pos_array), int(np.sqrt(len(pos_array))+1))
	bin_edges = midbin(bin_edges)
	y_err = np.sqrt(histo)
	x_err = data_in.d_px2um*np.sqrt(1+1/len(pos_array))

	#Approximate initial fitting parameters
	max_index = np.argmax(histo)
	max_counts = histo[max_index]
	max_pos = bin_edges[max_index]

	test_counts = max_counts
	upper_test_index = max_index
	while test_counts > max_counts/2:
		upper_test_index+=1
		test_counts = histo[upper_test_index]
	fwhm_hi = bin_edges[upper_test_index]

	test_counts = max_counts
	lower_test_index = max_index
	while test_counts > max_counts/2:
		lower_test_index-=1
		test_counts = histo[lower_test_index]
	fwhm_lo = bin_edges[lower_test_index]

	fwhm = fwhm_hi - fwhm_lo
	init_coeffs = [max_counts, max_pos, np.square(fwhm/2.35)]

	#Distinguish histogram used for fitting
	histo_fit = histo
	bin_edges_fit = bin_edges
	y_err_fit = y_err
	if len(histo)<=max_index+upper_test_index-lower_test_index:
		print("Heyup")
		histo_fit = histo_fit[:max_index+upper_test_index-lower_test_index]
		bin_edges_fit = bin_edges_fit[:max_index+upper_test_index-lower_test_index]
		y_err_fit = y_err_fit[:max_index+upper_test_index-lower_test_index]

	if max_index-upper_test_index+lower_test_index >= 0:
		print("Yooo")
		histo_fit = histo_fit[max_index-upper_test_index+lower_test_index:]
		bin_edges_fit = bin_edges_fit[max_index-upper_test_index+lower_test_index:]
		y_err_fit = y_err_fit[max_index-upper_test_index+lower_test_index:]

	zero_count_indices = []
	for i, count in enumerate(histo_fit):
		if count == 0:
			zero_count_indices.append(i)
	histo_fit = np.delete(histo_fit, zero_count_indices)
	bin_edges_fit = np.delete(bin_edges_fit, zero_count_indices)
	y_err_fit = np.delete(y_err_fit, zero_count_indices)

	#Orthogonal distance regression
	gauss = odr.Model(gaussian)
	mydata = odr.RealData(bin_edges_fit, histo_fit, sx=x_err, sy=y_err_fit)
	myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
	myoutput = myodr.run()

	return myoutput.beta, myoutput.sd_beta, histo, bin_edges, y_err, x_err

def posn_var(data_in, axis):
	pos_array = np.zeros((0,1))
	if axis == 'x' or axis == 'X':
		pos_array = data_in.x_pos
	elif axis == 'y' or axis == 'Y':
		pos_array = data_in.y_pos

	eq_pos = np.mean(pos_array)
	var = np.sum(np.square(pos_array- eq_pos))/len(pos_array)
	var*=data_in.px2um**2
	d_var = np.sqrt(8*var*(1+1/len(pos_array)))*data_in.d_px2um
	return var, d_var

def stiffness_calc(var, d_var):
	T, d_T = 293, 3
	stiffness = k*T/var
	stiffness*=10**12
	d_stiffness = stiffness*np.sqrt(np.square(d_T/T)+np.square(d_var/var))
	return stiffness, d_stiffness

dm.dir_data = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\lt121124"

axis = 'x'
infile = "test4_1_1_6_Camera_tr_Track"
outfile = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\LaserTweezerLab\\laserResults.txt"
data = dm.dataset(fname=infile)

#frame_diff = int(data.fps)+1

min_coeff, d_min_coeff, histo, bin_edges, y_err, x_err = fit_gaussian(data, axis)

zero_count_indices = []
for index, count in enumerate(histo):
	if count == 0:
		zero_count_indices.append(index)
histo_chi = np.delete(histo, zero_count_indices)
bin_edges_chi = np.delete(bin_edges, zero_count_indices)
y_err_chi = np.delete(y_err, zero_count_indices)

red_chi_sq = chi_square(histo_chi, gaussian(min_coeff, bin_edges_chi), y_err_chi)/(len(histo_chi)-3)

var, d_var = posn_var(data, axis)

stiff_1, d_stiff_1 = stiffness_calc(min_coeff[2], d_min_coeff[2])
stiff_2, d_stiff_2 = stiffness_calc(var, d_var)

print(f"For {axis} axis in file {infile}, the variances:\n\tGaussian fit:\t{min_coeff[2]} +/- {d_min_coeff[2]},\n\tCalculation:\t{var} +/- {d_var}\nStiffness:\n\tGaussian:\t{stiff_1}+/-{d_stiff_1}\n\tCalculation:\t{stiff_2}+/-{d_stiff_2}")

with open(outfile, "a") as file:
	file.write(f"{infile}\t{axis}\t{data.sphere_diameter/2}\t{data.sphere_volume}\t{data.water_volume}\t{data.fps}\t{data.exposure}\t{data.laser_current}\t{red_chi_sq}\t{min_coeff[2]}\t{d_min_coeff[2]}\t{stiff_1}\t{d_stiff_1}\t{var}\t{d_var}\t{stiff_2}\t{d_stiff_2}\n")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(bin_edges, histo, y_err, x_err, fmt="b.")
xax = np.linspace(bin_edges[0], bin_edges[-1], 400)
ax.plot(xax, gaussian(min_coeff, xax), "r")
ax.set_title(infile)
ax.set_ylabel("Counts")
ax.set_xlabel("Displacement from mean/um")
plt.show()
