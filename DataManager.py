import os
import numpy as np

dir_data = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\dataFiles"
exp_config = "C:\\Users\\veerk\\source\\repos\\LaserTweezers\\LaserTweezerLab\\experimentConfig.txt"

def chi_square(true_val, test_val, std):
	return np.sum(np.square((test_val - true_val)/std))

def midbin(bin_edges):
	midpoints = np.zeros((0,1))
	last_edge = bin_edges[0]
	for next_edge in bin_edges[1:]:
		midpoints = np.append(midpoints, (next_edge + last_edge)/2)
		last_edge = next_edge
	return midpoints

def msd_histo_1d(pos_arr, n_frames, px2um):
	#Outputs histogram and bin edges of msd distribution; note that uncertainty in hostogram counts = sqrt(counts)
	buffer = pos_arr[:n_frames]
	msd = np.zeros((0,1))
	for pos in pos_arr[n_frames:]:
		msd = np.vstack((msd, px2um*(pos - buffer[0])))
		buffer = np.append(buffer[1:], pos)
	#max_error = 2*np.sqrt(2*max(msd))*d_px2um
	#max_error = np.sqrt(2)*d_px2um
	#n_bins = int(np.ceil((max(msd)[0] - min(msd)[0])/max_error))
	#histo, bin_edges = np.histogram(msd, n_bins, density=False)
	histo, bin_edges = np.histogram(msd, int(np.sqrt(len(msd)))+1, density=False)
	bin_edges = midbin(px2um*bin_edges)
	return histo, bin_edges

def cumulative_travel_1d(pos_arr, n_frames, fps, px2um):
	#Outputs cumulative travel distance and time; note that uncertainty in the nth cumulative distance = sqrt(n)*(error in one position measurement) 
	buffer = pos_arr[:n_frames]
	dist_arr = np.zeros((1,1))
	for pos in pos_arr[n_frames:]:
		dist_arr = np.append(dist_arr, dist_arr[-1]+abs(pos - buffer[0]))
		buffer = np.append(buffer[1:], pos)
	dist_arr = dist_arr[1:]*px2um
	time_arr = np.linspace(0, len(dist_arr)-1, len(dist_arr))/fps
	return dist_arr, time_arr


def msd_calc_1d(pos_arr, n_frames, px2um, d_px2um):
	buffer = pos_arr[:n_frames]
	add = 0.0
	for pos in pos_arr[n_frames:]:
		add += (pos - buffer[0])**2
		buffer = np.vstack((buffer[1:], pos))
	msd = px2um**2 * add / (len(pos_arr) - n_frames)
	return msd[0], d_px2um*np.sqrt((8*msd)/(len(pos_arr) - n_frames))[0]
	#return msd[0], np.sqrt(d_px2um*(8*msd))[0]

class datasetError(Exception):
	def __init__(self, msg):
		self.message = msg
		super().__init__(self.message)

class dataset:
	def __init__(self, day_index=0, sphere_index=0, conc_index=0, rec_index=0, particle_index=0, fname='filename'):
		self.file = fname
		self.sphere_diameter = 0.0 #um
		self.sphere_volume = 0.0 #uL
		self.water_volume = 0.0 #mL
		self.roi = [0, 0] #[x, y]
		self.fps = 0.0
		self.exposure = 0.0 #ms
		self.px2um = 0.0 #um
		self.d_px2um = 0.0 #um
		self.laser_current = 0.0 #mA
		self.particles_tracked = 0

		self.x_pos = np.zeros((0,1))
		self.y_pos = np.zeros((0,1))
		self.frames = np.zeros((0,1))

		self.disp_histo_counts = np.zeros((0,1))
		self.disp_histo_pos = np.zeros((0,1))
		self.cumulative_distance = np.zeros((0,1))
		self.cumulative_time = np.zeros((0,1))
		self.msd = 0.0
		self.d_msd = 0.0

		self.valid_instance = True
		
		if type(day_index) != int or type(sphere_index) != int or type(conc_index) != int or type(rec_index) != int or type(fname) != str:
			self.valid_instance = False
			raise datasetError("Incorrectly formatted dataset parameters")

		if day_index != 0 and sphere_index != 0 and conc_index != 0 and rec_index != 0:
			#Index input, particle number assigned to first file if not specified (left at 0), if index is too high an error is thrown
			with open(exp_config) as file:
				for line in file:
					line = line.strip("\n").split("\t")
					try:
						if int(line[0]) == day_index and int(line[1]) == sphere_index and int(line[3]) == conc_index and int(line[6]) == rec_index:
							self.sphere_diameter = float(line[2])
							self.sphere_volume = float(line[4])
							self.water_volume = float(line[5])
							self.roi = [int(line[7]), int(line[8])]
							self.fps = float(line[9])
							self.exposure = float(line[10])
							self.px2um = float(line[11])
							self.d_px2um = float(line[12])
							self.laser_current = float(line[13])
							break
					except (ValueError, TypeError):
						pass
				if self.sphere_diameter == 0.0:
					self.valid_instance = False
					raise datasetError("Inputted indices do not match any dataset, try specifying by filename")
			
			self.file = f"test{day_index}_{sphere_index}_{conc_index}_{rec_index}_Camera_tr_Track"
			files_from_rec = []
			for file in os.listdir(dir_data):
				if file.split(".")[0] == self.file:
					files_from_rec.append(file)
			self.particles_tracked = len(files_from_rec)

			if particle_index<self.particles_tracked:
				self.file = files_from_rec[particle_index-1]
			else:
				self.valid_instance = False
				if self.particles_tracked == 0:
					raise datasetError(f"No data files found in directory: {dir_data}")
				else:
					raise datasetError(f"Recording does not track more than {self.particles_tracked} particles; asked for more than that")

		else:
			#Test filename, throw error if invalid, otherwise retrieve metadata from config file, throw error if not logged
			if fname == fname.strip(".csv"):
				self.file = fname+".csv"
			else:
				self.file = fname

			os.chdir(dir_data)
			if not os.path.isfile(self.file):
				self.valid_instance = False
				raise datasetError(f"File ({self.file}) could not be found in data directory ({dir_data})")
			for file in os.listdir(dir_data):
				if file.split(".")[0] == fname.split(".")[0]:
					self.particles_tracked+=1
			
			indices = [] #Day, Sphere index, Concentration index, Recording index, Particle in recording
			for i in fname.lstrip("test").split("_")[:-3]:
				indices.append(int(i))
			if fname.strip(".csv").split(".")[-1].isdigit():
				indices.append(int(fname.strip(".csv").split(".")[-1].isdigit()))
			else:
				indices.append(0)

			with open(exp_config) as file:
				for line in file:
					line = line.strip("\n").split("\t")
					try:
						if int(line[0]) == indices[0] and int(line[1]) == indices[1] and int(line[3]) == indices[2] and int(line[6]) == indices[3]:
							self.sphere_diameter = float(line[2])
							self.sphere_volume = float(line[4])
							self.water_volume = float(line[5])
							self.roi = [int(line[7]), int(line[8])]
							self.fps = float(line[9])
							self.exposure = float(line[10])
							self.px2um = float(line[11])
							self.d_px2um = float(line[12])
							self.laser_current = float(line[13])
							break
					except (ValueError, TypeError):
						pass
				if self.sphere_diameter == 0.0:
					self.valid_instance = False
					raise datasetError(f"Metadata for file \'{self.file}' could not be found in experiment config log \'{exp_config}'")


		os.chdir(dir_data)
		with open(self.file) as file:
			for line in file:
				line = line.strip("\n").split(",")
				self.frames = np.vstack((self.frames, int(line[0])))
				self.x_pos = np.vstack((self.x_pos, float(line[1])))
				self.y_pos = np.vstack((self.y_pos, float(line[2])))

	#Not averaging anything
	def displacement_histo(self, n, axis = 'x'):
		'''
		n = integer no. frames before current frame to compare with current frame
		'''
		if axis == 'X' or axis == 'x':
			self.disp_histo_counts, self.disp_histo_pos = msd_histo_1d(self.x_pos, n, self.px2um)
		elif axis == 'Y' or axis == 'y':
			self.disp_histo_counts, self.disp_histo_pos = msd_histo_1d(self.y_pos, n, self.px2um)
		'''
		#Should produce 2D histogram data - work out formatting with class structure
		else:
			x_buffer = self.x_pos[:n]
			y_buffer = self.y_pos[:n]
			msd = np.zeros((0,1))
			for x_pos, y_pos in zip(self.x_pos[n:], self.y_pos[n:]):
				msd = np.vstack((msd, np.square(x_pos - x_buffer[0])+np.square(y_pos - y_buffer[0])))
				x_buffer, y_buffer = np.append(x_buffer[:1], x_pos), np.append(y_buffer[:1], y_pos)
			max_error = 2*np.sqrt(2*max(msd))*d_px2um
			n_bins = int(np.ceil((max(msd) - min(msd))/np.sqrt(2*d_px2um)))
			histo, bin_edges = np.histogram(msd, n_bins, density=False)
			bin_edges = px2um*bin_edges
			return histo, bin_edges
		'''
	def cumulative_travel(self, n, axis):
		if axis == "X" or axis == "x":
			self.cumulative_distance, self.cumulative_time = cumulative_travel_1d(self.x_pos, n, self.fps, self.px2um)
		elif axis == "Y" or axis == "y":
			self.cumulative_distance, self.cumulative_time = cumulative_travel_1d(self.y_pos, n, self.fps, self.px2um)
		else:
			x_buffer, y_buffer = self.x_pos[:n], self.y_pos[:n]
			dist_arr = np.zeros((1,1))
			for x_pos, y_pos in zip(self.x_pos[n:], self.y_pos[n:]):
				dist_arr = np.append(dist_arr, dist_arr[-1]+np.sqrt(np.square(x_pos - x_buffer[0])+np.square(y_pos - y_buffer[0])))
				x_buffer, y_buffer = np.append(x_buffer[:1], x_pos), np.append(y_buffer[:1], y_pos)
			self.cumulative_distance = dist_arr[1:]*self.px2um
			self.cumulative_time = np.linspace(0, len(dist_arr)-1, len(dist_arr))/self.fps

	def msd_calc(self, n, axis):
		if axis == 'X' or axis == 'x':
			self.msd, self.d_msd = msd_calc_1d(self.x_pos, n, self.px2um, self.d_px2um)
		elif axis == 'Y' or axis == 'y':
			self.msd, self.d_msd = msd_calc_1d(self.y_pos, n, self.px2um, self.d_px2um)
		else:
			x_msd, x_d_msd = msd_calc_1d(self.x_pos, n, self.px2um, self.d_px2um)
			y_msd, y_d_msd = msd_calc_1d(self.y_pos, n, self.px2um, self.d_px2um)
			self.msd = x_msd + y_msd
			self.d_msd = (y_d_msd**2 + x_d_msd**2)*0.5