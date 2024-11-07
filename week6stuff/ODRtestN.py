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

bad_files = {}

class recording:
	def __init__(self, day_index, sphere_index, conc_index, rec_index):
		#indices
		self.day_index = day_index
		self.sphere_index = sphere_index
		self.conc_index = conc_index
		self.rec_index = rec_index

		#metadata of recording
		data = dm.dataset(self.day_index, self.sphere_index, self.conc_index, self.rec_index)
		self.diameter = data.sphere_diameter
		self.sphere_vol = data.sphere_volume
		self.water_vol = data.water_volume
		self.fps = data.fps
		self.exposure = data.exposure
		self.n_particles = data.particles_tracked

		#viscosity data
		self.viscosity = {"x": [], "y": []}
		self.d_viscosity = {"x": [], "y": []}
		self.filenames = []

	def generate_data(self, max_delay=1):
		'''max_delay = maximum time delay between frames in seconds'''
		frame_diff_start = 2
		frame_diff_end = int(max_delay*self.fps)+1
		frame_diff_interval = 2

		for particle in range(self.n_particles):
			data = dm.dataset(self.day_index, self.sphere_index, self.conc_index, self.rec_index, particle)
			try:
				for axis in ["x", "y"]:
					fit_coeffs = np.zeros((0,3))
					fit_coeffs_unc = np.zeros((0,3))
					delays = np.zeros((0,1))

					for frame_diff in range(frame_diff_start, frame_diff_end, frame_diff_interval):
						#generate histogram of displacements after frame_diff frames
						data.displacement_histo(frame_diff, axis)
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

						#USe orthogonal distance regression to 
						gauss = odr.Model(gaussian)
						#mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=unc, sy=np.sqrt(data.disp_histo_counts))
						mydata = odr.RealData(data.disp_histo_pos, data.disp_histo_counts, sx=data.d_px2um*np.sqrt(2)) #Are we determining the error from binning random variables correctly?
						myodr = odr.ODR(mydata, gauss, beta0=init_coeffs)
						myoutput = myodr.run()

						fit_coeffs = np.vstack((fit_coeffs, myoutput.beta))
						fit_coeffs_unc = np.vstack((fit_coeffs_unc, myoutput.sd_beta))
						delays = np.append(delays, frame_diff/data.fps)

					#Fit linear function to variance vs delay
					#print(fit_coeffs)
					init_grad = (fit_coeffs[-1,2]- fit_coeffs[0,2])/(delays[-1]-delays[0])
					minimisation = fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]), np.array([0, init_grad]), full_output=True, disp=False)
					coeff_min, chi_min = minimisation[0], minimisation[1]

					offset_b = np.power(10, np.floor(np.log10(coeff_min[1])-1))
					min_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[1], coeff_min - [0, offset_b], full_output=False, disp=False)[1]
					max_b = fmin(lambda coeff_min_test: fmin(lambda coeffs: chi_square(fit_coeffs[:,2], linear(coeffs, delays), fit_coeffs_unc[:,2]) - chi_min - 1, coeff_min_test, full_output=False, disp=False)[1], coeff_min + [0, offset_b], full_output=False, disp=False)[1]
					std_b = 0.5*abs(max_b-min_b)
					viscosity = 293*k*10**18/(1.5*pi*self.diameter*coeff_min[1])
					d_viscosity = viscosity*std_b/coeff_min[1]

					self.viscosity[axis].append(viscosity)
					self.d_viscosity[axis].append(d_viscosity)
				self.filenames.append(data.file)
			except Exception as err:
				bad_files[data.file] = err

recordings = []
with open (dm.exp_config) as file:
	for line in file:
		try:
			line = line.strip("\n").split("\t")
			if int(line[13])==0:
				rec = recording(int(line[0]), int(line[1]), int(line[3]), int(line[6]))
				rec.generate_data()
				recordings.append(rec)
				print(f"Files: {rec.filenames}")
				print(f"Viscosities: {rec.viscosity}")
				print(f"Std viscosities: {rec.d_viscosity}")

		except (TypeError, ValueError):
			pass

print(bad_files)


