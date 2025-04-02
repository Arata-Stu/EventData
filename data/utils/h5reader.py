import h5py
try:
    import hdf5plugin
except ImportError:
    pass

import numpy as np
import torch
from numba import jit
from pathlib import Path
import weakref
from typing import Tuple

from utils.timers import Timer

class H5Reader:
    def __init__(self, h5_file: Path, width: int = None, height: int = None):
        assert h5_file.exists(), f"{h5_file} does not exist."
        assert h5_file.suffix in ['.hdf5', '.h5'], "File must be HDF5 format."
        
        self.h5f = h5py.File(str(h5_file), 'r')  # HDF5ファイルを開く
        self._finalizer = weakref.finalize(self, self._close_callback, self.h5f)
        self.is_open = True

        # 高さと幅を設定
        self.width, self.height = width, height
        self.all_times = None

        # ファイル構造を判定
        if "CD" in self.h5f and "events" in self.h5f["CD"]:
            self.event_path = ["CD", "events"]
        elif "events" in self.h5f:
            self.event_path = ["events"]
        else:
            raise ValueError("Unsupported H5 file structure. Cannot find events data.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()

    @staticmethod
    def _close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self.h5f.close()
        self.is_open = False

    def get_height_and_width(self) -> Tuple[int, int]:
        """
        高さと幅を返す。明示的に設定されていない場合は None を返す。
        """
        return self.height, self.width

    @property
    def time(self) -> np.ndarray:
        """
        "t"のデータを遅延的にロードして、時系列を補正して返す。
        """
        assert self.is_open, "File is closed."
        if self.all_times is None:
            event_group = self._get_event_group()
            self.all_times = np.asarray(event_group['t'])
            self._correct_time(self.all_times)
        return self.all_times

    @staticmethod
    @jit(nopython=True)
    def _correct_time(time_array: np.ndarray):
        """
        タイムスタンプが降順になる場合に補正。
        """
        assert time_array[0] >= 0, "Time must start from non-negative values."
        time_last = 0
        for idx, tval in enumerate(time_array):
            if tval < time_last:
                time_array[idx] = time_last
            else:
                time_last = tval
    
    def get_original_dtypes(self) -> dict:
        """
        t, x, y, p それぞれの要素が元々どのデータ型で保存されていたかを取得する。
        Returns:
            dict: 各データの元のデータ型を示す辞書
        """
        assert self.is_open, "File is closed."
        event_group = self._get_event_group()
        dtypes = {
            "t": event_group['t'].dtype,
            "x": event_group['x'].dtype,
            "y": event_group['y'].dtype,
            "p": event_group['p'].dtype
        }
        return dtypes

    def get_event_slice(self, idx_start: int, idx_end: int, convert_2_torch: bool = False) -> dict:
        """
        一部分のイベントを取得する。
        """
        assert self.is_open
        assert idx_end >= idx_start

        event_group = self._get_event_group()
        x_array = np.asarray(event_group['x'][idx_start:idx_end], dtype='int64')
        y_array = np.asarray(event_group['y'][idx_start:idx_end], dtype='int64')
        p_array = np.asarray(event_group['p'][idx_start:idx_end], dtype='int64')
        p_array = np.clip(p_array, a_min=0, a_max=None)
        t_array = np.asarray(self.time[idx_start:idx_end], dtype='int64')

        # タイムスタンプが昇順か確認
        assert np.all(t_array[:-1] <= t_array[1:])

        ev_data = dict(
            x=x_array if not convert_2_torch else torch.from_numpy(x_array),
            y=y_array if not convert_2_torch else torch.from_numpy(y_array),
            p=p_array if not convert_2_torch else torch.from_numpy(p_array),
            t=t_array if not convert_2_torch else torch.from_numpy(t_array),
            height=self.height,
            width=self.width,
        )
        return ev_data

    def get_event_summary(self) -> dict:
        """
        イベントデータ全体の統計情報を返す。
        - タイムスタンプの最小値・最大値
        - x, y の最小値・最大値
        - ON/OFF (p=1, p=0) の数
        - 全イベント数
        - 読み込みにかかった時間
        """
        assert self.is_open, "File is closed."

        with Timer(timer_name="get_event_summary"):
            # イベントデータを取得
            event_group = self._get_event_group()
            x_array = np.asarray(event_group['x'])
            y_array = np.asarray(event_group['y'])
            p_array = np.asarray(event_group['p'])
            t_array = self.time  # self.timeはすでに補正済みのタイムスタンプを返す

            t_min = t_array.min()
            t_max = t_array.max()
            x_min = x_array.min()
            x_max = x_array.max()
            y_min = y_array.min()
            y_max = y_array.max()

            p_on_count = np.count_nonzero(p_array == 1)
            p_off_count = np.count_nonzero(p_array == 0)
            total_count = len(p_array)

            summary = {
                "t_min": t_min,
                "t_max": t_max,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "p_on_count": p_on_count,
                "p_off_count": p_off_count,
                "total_count": total_count,
            }
        
        return summary


    def _get_event_group(self):
        """
        イベントデータが格納されているグループを取得する。
        """
        group = self.h5f
        for key in self.event_path:
            group = group[key]
        return group
