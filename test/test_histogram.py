import sys
sys.path.append("../")

import numpy as np
import torch as th
import argparse
from pathlib import Path
from data.utils.h5reader import H5Reader
from data.utils.representation import StackedHistogram
from utils.timers import Timer

def main(h5_file_path, width, height, delta_t_ms):
    delta_t_us = delta_t_ms * 1000
    h5_file = Path(h5_file_path)
    reader = H5Reader(h5_file, width=width, height=height)
    height, width = reader.get_height_and_width()
    event_histogram = StackedHistogram(bins=2, height=height, width=width, count_cutoff=10, downsample=False)

    all_time = reader.time
    idx_start = 0
    print(f"Total events: {len(all_time)}")
    
    while idx_start < len(all_time):
        with Timer(timer_name="EventHistogram"):
            # delta_t_ns分のイベントをスライス
            idx_end = np.searchsorted(all_time, all_time[idx_start] + delta_t_us, side='right')
            events = reader.get_event_slice(idx_start, idx_end, convert_2_torch=False)
            idx_start = idx_end

            # Numpy配列をTensorに変換
            t = events['t']
            x = events['x']
            y = events['y']
            p = events['p']

            # イベントデータをフレームに変換
            histogram = event_histogram.construct(x=x, y=y, pol=p, time=t)

            # # フレーム形式に応じた処理 (CHW → HWC)
            # if isinstance(frame, th.Tensor):
            #     frame = frame.permute(1, 2, 0).cpu().numpy()  # Tensor を numpy に変換
            # elif isinstance(frame, np.ndarray):
            #     frame = frame.transpose(1, 2, 0)  # Numpy 形式の次元を変更
            # else:
            #     raise ValueError("Unsupported frame format. Frame must be either Tensor or Numpy array.")
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 events to images.")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--width", type=int, required=True, help="Camera Width.")
    parser.add_argument("--height", type=int, required=True, help="Camera Height.")
    parser.add_argument("--delta_t_ms", type=int, required=True, default=100, help="Time interval in ms for event slices.")
    args = parser.parse_args()

    main(args.h5_file, args.width, args.height, args.delta_t_ms)
