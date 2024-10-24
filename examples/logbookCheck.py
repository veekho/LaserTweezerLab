#Check experimentConfigLog for any discarded datasets
import os
import DataManager as dm

expIndices = []

with open(dm.exp_config) as file:
	for line in file:
		try:
			line = line.strip("\n").split("\t")
			expIndices.append([int(line[0]), int(line[1]), int(line[3]), int(line[6])])
		except (TypeError, ValueError):
			pass

for line, indices in enumerate(expIndices):
	try:
		data = dm.dataset(indices[0], indices[1], indices[2], indices[3])
		if data.valid_instance == True:
			print(f"Data FOUND for line {line+2}, with indices: {indices}\n\t{data.particles_tracked} recordings")
	except dm.datasetError as err:
		print(f"Data MISSING for line {line+2}, with indices: {indices} \n\t{err}")

print(f"\n\nCompleted read through experiment log file: {dm.exp_config}\nNow checking through data director: {dm.dir_data}\n\n")

file_set = os.listdir(dm.dir_data)

for file in file_set:
	try:
		filename = file.lstrip("test").split("_")
		data = dm.dataset(int(filename[0]), int(filename[1]), int(filename[2]), int(filename[3]))
		if data.valid_instance:
			print(f"NO ISSUE with data under filename: {file}")
	except (ValueError, TypeError, IndexError) as err:
		print(f"PROBLEM formatting in the filename: {file}\n\t{err}")
	except dm.datasetError as err:
		print(f"PROBLEM finding dataset in experiment config\n\t{err}")