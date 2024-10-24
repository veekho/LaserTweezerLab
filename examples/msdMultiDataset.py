#This code is just an excercise in comparing results across multiple datasets
#Will compare the MSD for all measurements concerning sphere size 2 on day 1 - datasets across varying concentration, exposure/fps and for multiple particles per recording
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import DataManager as dm


#Adjust these as you wish
sphere_index = 2
conc_index = 1
no_recordings = 5

#Will generate a series of MSD vs Delay time plots, over all exposure & fps settings
#Therefore the number of MSD's calculated/Delay time intervals will be determines by fps
max_sample_period = 0.5 #seconds
#Will do this for x, y and x+y
#Can use a cmap in the plot to distinguish between x/y/x+y plots, exposure/fps settings, 
#Will not fit data

color_mode = "exposure" #"axis", "fps", "exposure"
delay = []
fps = []
exposure = []

msd_x = []
msd_y = []
msd_xy = []
d_msd_x = []
d_msd_y = []
d_msd_xy = []

for recording in range(1, no_recordings+1):
	particle_index = 0
	total_particles_tracked = 1
	while particle_index < total_particles_tracked:
		data = dm.dataset(1, sphere_index, conc_index, recording)
		total_particles_tracked = data.particles_tracked
		no_frame_intervals = int(max_sample_period*data.fps)

		for frame_interval in range(1, no_frame_intervals+1):
			data.msd_calc(frame_interval, "x")
			msd_x.append(data.msd)
			d_msd_x.append(data.d_msd)

			data.msd_calc(frame_interval, "y")
			msd_y.append(data.msd)
			d_msd_y.append(data.d_msd)
			
			data.msd_calc(frame_interval, "xy")
			msd_xy.append(data.msd)
			d_msd_xy.append(data.d_msd)
			
			delay.append(frame_interval/data.fps)
			fps.append(data.fps)
			exposure.append(data.exposure)


		particle_index+=1


fig = plt.figure()
ax=None


if color_mode == "axis":
	ax = fig.add_subplot(111)
	ax.errorbar(delay, msd_x, d_msd_x, fmt="b.", label="X")
	ax.errorbar(delay, msd_y, d_msd_y, fmt="r.", label="Y")
	ax.errorbar(delay, msd_xy, d_msd_xy, fmt="g.", label="X+Y")
	
	fig.legend(loc="upper right")
else:
	ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])
	cax = fig.add_axes([0.78, 0.1, 0.05, 0.8])
	
	cmap = mpl.cm.plasma
	norm = None
	cax_label = None
	col = None
	ticks = None
	if color_mode == "fps":
		ticks = [min(fps), max(fps)]
		norm = mpl.colors.Normalize(vmin=ticks[0], vmax=ticks[1])
		col = cmap(norm(fps))
		cax_label = "Frame rate/FPS"
	elif color_mode == "exposure":
		ticks = [min(exposure), max(exposure)]
		norm = mpl.colors.Normalize(vmin=ticks[0], vmax=ticks[1])
		col = cmap(norm(exposure))
		cax_label = "Exposure time/ms"

	sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
	cb = fig.colorbar(sm, cax=cax, ticks=ticks, label=cax_label)
	cb.set_ticklabels([f"{ticks[0]:.1f}", f"{ticks[1]:.1f}"])
	
	for t, x, dx, y, dy, xy, dxy, c in zip(delay, msd_x, d_msd_x, msd_y, d_msd_y, msd_xy, d_msd_xy, col):
		ax.errorbar(t, x, dx, color=c, marker=".")
		ax.errorbar(t, y, dy, color=c, marker=".")
		ax.errorbar(t, xy, dxy, color=c, marker=".")

ax.set_xlabel("Delay/s")
ax.set_ylabel("MSD")

plt.show()