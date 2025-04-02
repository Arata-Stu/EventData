import sys
sys.path.append("..")

from pathlib import Path
from data.utils.h5reader import H5Reader

h5_path = Path("/Users/at/Downloads/driving_sample.hdf5") 
# h5_path = Path("/Users/at/dataset/moorea_2019-01-30_000_td_671500000_731500000_td.h5")
width, height = 640, 360
h5_reader = H5Reader(h5_file=h5_path, width=width, height=height)

print(h5_reader.get_height_and_width())
print(h5_reader.get_original_dtypes())
print(h5_reader.get_event_summary())
