from typing import Tuple, Union, Optional

import numpy as np
import torch as th
import cv2


from abc import ABC, abstractmethod
import numba as nb

@nb.njit
def create_frame_jit(x, y, pol, height, width):
    """
    Numba による JIT 化版 majority vote 関数。
    ループを用いて ON/OFF のカウントを行い、最終的なピクセル値を決定する。
    """
    # 各ピクセル位置の ON/OFF カウント (1次元配列)
    counts_on = np.zeros(height * width, dtype=np.int32)
    counts_off = np.zeros(height * width, dtype=np.int32)
    n = x.shape[0]
    
    for i in range(n):
        # 座標のクリッピング
        xi = x[i]
        if xi < 0:
            xi = 0
        elif xi >= width:
            xi = width - 1
        
        yi = y[i]
        if yi < 0:
            yi = 0
        elif yi >= height:
            yi = height - 1
        
        idx = yi * width + xi
        
        # イベントのカウント（pol==1: ON, pol==0: OFF）
        if pol[i] == 1:
            counts_on[idx] += 1
        elif pol[i] == 0:
            counts_off[idx] += 1
    
    # 出力フレームを 1 次元配列で生成
    frame = np.empty(height * width, dtype=np.uint8)
    for i in range(height * width):
        diff = counts_on[i] - counts_off[i]
        if diff > 0:
            frame[i] = 255
        elif diff < 0:
            frame[i] = 0
        else:
            frame[i] = 127
    return frame.reshape(height, width)


@nb.njit
def compute_histogram_jit(x, y, time, pol, bins, height, width, count_cutoff):
    n = x.shape[0]
    # チャンネルは 2 として、int32 で集計
    rep = np.zeros((2, bins, height, width), dtype=np.int32)
    t0 = time[0]
    t1 = time[-1]
    dt = t1 - t0
    if dt < 1:
        dt = 1  # ゼロ除算回避
    for i in range(n):
        # 時間の正規化と bin インデックスの計算
        t_norm = (time[i] - t0) / dt
        t_idx = int(t_norm * bins)
        if t_idx >= bins:
            t_idx = bins - 1

        # 座標のクリッピング
        xi = x[i]
        if xi < 0:
            xi = 0
        elif xi >= width:
            xi = width - 1

        yi = y[i]
        if yi < 0:
            yi = 0
        elif yi >= height:
            yi = height - 1

        # チャンネル（pol: 0または1）を決定してカウントを加算
        c = int(pol[i])
        rep[c, t_idx, yi, xi] += 1

    # 各画素で count_cutoff を超えた部分をクリップ
    for c in range(2):
        for b in range(bins):
            for i in range(height):
                for j in range(width):
                    if rep[c, b, i, j] > count_cutoff:
                        rep[c, b, i, j] = count_cutoff
    return rep

class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...


class EventFrame(RepresentationBase):
    def __init__(self, height: int, width: int, downsample: bool = False):
        """
        Majority vote based Event Frame representation.
        各ピクセルにおいて、ON と OFF イベントの多数決を行いRGB画像に変換する。
        ONイベントが多ければ白（255）、OFFイベントが多ければ黒（0）、同数なら灰色（127）に設定する。
        :param height: フレームの高さ
        :param width: フレームの幅
        :param downsample: ダウンサンプリングを行うか（半分に縮小）
        """
        super().__init__()
        self.height = height
        self.width = width
        self.downsample = downsample

    def get_shape(self) -> Tuple[int, int, int]:
        # RGB画像の形状
        if self.downsample:
            return (3, self.height // 2, self.width // 2)
        return (3, self.height, self.width)
    
    def create_frame_tensor(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        Majority vote によるイベントフレームを作成する (PyTorch版)。
        各ピクセルで ON と OFF のイベント数をカウントし、多数決を行う。
        :param x: x 座標
        :param y: y 座標
        :param pol: イベントの極性 (+1 for ON, 0 for OFF)
        :param time: タイムスタンプ（未使用）
        :return: RGB画像 (torch.Tensor)
        """
        device = x.device
        # 座標をフレームサイズ内にクリップ
        x_clipped = th.clamp(x, min=0, max=self.width - 1)
        y_clipped = th.clamp(y, min=0, max=self.height - 1)
        
        # ON/OFF のマスク作成
        on_mask = (pol == 1)
        off_mask = (pol == 0)
        
        # 1次元のインデックスに変換：idx = y * width + x
        idx_on = y_clipped[on_mask] * self.width + x_clipped[on_mask]
        idx_off = y_clipped[off_mask] * self.width + x_clipped[off_mask]
        
        # torch.bincount により、各ピクセルのカウントを集計（出力は1次元なので reshape する）
        count_on = th.bincount(idx_on, minlength=self.height * self.width).reshape(self.height, self.width)
        count_off = th.bincount(idx_off, minlength=self.height * self.width).reshape(self.height, self.width)
        
        # 差分計算：多数決（diff > 0: ON 多, diff < 0: OFF 多, diff == 0: 同数）
        diff = count_on - count_off
        
        # 単一チャンネルのフレーム作成：初期値 127（灰色）
        frame = th.full((self.height, self.width), 127, dtype=th.uint8, device=device)
        frame[diff > 0] = 255  # ON 多 -> 白
        frame[diff < 0] = 0    # OFF 多 -> 黒
        
        # 3 チャンネル画像に変換
        img = th.stack([frame, frame, frame], dim=0)
        
        # ダウンサンプリングが必要な場合はリサイズ
        if self.downsample:
            img = th.nn.functional.interpolate(
                img.unsqueeze(0).float(),
                size=(self.height // 2, self.width // 2),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).to(th.uint8)
        
        return img


    
    def create_frame_numpy(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Majority vote によるイベントフレームを作成する (NumPy版＋Numba JIT)。
        JIT 化した関数を用いて各ピクセルで ON と OFF のイベント数をカウントし、多数決を行う。
        
        :param x: x 座標
        :param y: y 座標
        :param pol: イベントの極性 (+1 for ON, 0 for OFF)
        :param time: タイムスタンプ（未使用）
        :return: RGB画像 (numpy.ndarray)
        """
        # JIT 化した関数で majority vote 結果（2D フレーム）を取得
        frame = create_frame_jit(x, y, pol, self.height, self.width)
        # 3チャンネル画像に変換
        img = np.stack([frame] * 3, axis=0)
        
        # ダウンサンプリングが必要な場合はリサイズ
        if self.downsample:
            new_height = self.height // 2
            new_width = self.width // 2
            img_resized = np.zeros((3, new_height, new_width), dtype=np.uint8)
            for c in range(3):
                # cv2.resize の引数は (width, height) の順
                img_resized[c] = cv2.resize(img[c], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            img = img_resized
        
        return img

    def construct(self,
                  x: Union[th.Tensor, np.ndarray],
                  y: Union[th.Tensor, np.ndarray],
                  pol: Union[th.Tensor, np.ndarray],
                  time: Union[th.Tensor, np.ndarray]) -> Union[th.Tensor, np.ndarray]:
        """
        入力の型に応じて、PyTorch または NumPy 版のイベントフレームを作成する。
        :param x: x 座標
        :param y: y 座標
        :param pol: イベントの極性 (+1 for ON, 0 for OFF)
        :param time: タイムスタンプ（未使用）
        :return: RGB画像 (torch.Tensor または numpy.ndarray)
        """
        if isinstance(x, th.Tensor):
            return self.create_frame_tensor(x, y, pol, time)
        elif isinstance(x, np.ndarray):
            return self.create_frame_numpy(x, y, pol, time)
        else:
            raise ValueError("Unsupported type for input data.")

        

class StackedHistogram(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None, fastmode: bool = True, downsample: bool = False):
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = 255 if count_cutoff is None else min(count_cutoff, 255)
        self.fastmode = fastmode
        self.channels = 2
        self.downsample = downsample

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: np.ndarray):
        # representation の shape: (channels, bins, height, width)
        return representation.reshape((-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        if self.downsample:
            return (2 * self.bins, self.height // 2, self.width // 2)
        return (2 * self.bins, self.height, self.width)

    def create_from_numpy(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        assert x.shape == y.shape == pol.shape == time.shape
        dtype = np.uint8 if self.fastmode else np.int16

        # イベントが無い場合は空のヒストグラムを返す
        if x.size == 0:
            representation = np.zeros((self.channels, self.bins, self.height, self.width), dtype=dtype)
            return self.merge_channel_and_bins(representation.astype(np.uint8))

        # JIT 化した関数を用いてヒストグラムを計算
        rep = compute_histogram_jit(x, y, time, pol, self.bins, self.height, self.width, self.count_cutoff)
        # fastmode の場合は uint8 に変換
        if self.fastmode:
            rep = rep.astype(np.uint8)
        else:
            rep = rep.astype(np.int16)

        # ダウンサンプリングが必要な場合は cv2.resize を利用
        if self.downsample:
            new_height = self.height // 2
            new_width = self.width // 2
            rep_resized = np.zeros((self.channels, self.bins, new_height, new_width), dtype=np.uint8)
            for c in range(self.channels):
                for b in range(self.bins):
                    rep_resized[c, b] = cv2.resize(rep[c, b], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            rep = rep_resized
            out_shape = (2 * self.bins, new_height, new_width)
        else:
            out_shape = (2 * self.bins, self.height, self.width)

        return rep.reshape(out_shape)

    def create_frame_tensor(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        # こちらは PyTorch 版の処理です。JIT 化の対象は主に numpy 版で検証するという前提です。
        device = x.device
        assert y.device == pol.device == time.device == device
        dtype = th.uint8 if self.fastmode else th.int16

        representation = th.zeros((self.channels, self.bins, self.height, self.width),
                                  dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            return self.merge_channel_and_bins(representation.to(th.uint8))

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width
        t0_int = time[0]
        t1_int = time[-1]
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_idx = th.clamp((t_norm * bn).floor(), max=bn - 1).long()

        indices = x.long() + wd * y.long() + ht * wd * t_idx + bn * ht * wd * pol.long()
        values = th.ones_like(indices, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = th.clamp(representation, min=0, max=self.count_cutoff)

        if self.downsample:
            representation = th.nn.functional.interpolate(
                representation.float().unsqueeze(0),
                size=(self.channels, self.bins, self.height // 2, self.width // 2),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).to(th.uint8)

        return self.merge_channel_and_bins(representation)

    def construct(self,
                  x: Union[th.Tensor, np.ndarray],
                  y: Union[th.Tensor, np.ndarray],
                  pol: Union[th.Tensor, np.ndarray],
                  time: Union[th.Tensor, np.ndarray]) -> Union[th.Tensor, np.ndarray]:
        if isinstance(x, th.Tensor):
            return self.create_frame_tensor(x, y, pol, time)
        elif isinstance(x, np.ndarray):
            return self.create_from_numpy(x, y, pol, time)
        else:
            raise ValueError("Unsupported type for input data.")

def cumsum_channel(x: th.Tensor, num_channels: int):
    for i in reversed(range(num_channels)):
        x[i] = th.sum(input=x[:i + 1], dim=0)
    return x


class TimeSurface(RepresentationBase):
    def __init__(self, height: int, width: int, decay_const: float, downsample: bool = False):
        """
        Time Surface representation that encodes the last event times in a decaying format.
        :param height: Height of the time surface.
        :param width: Width of the time surface.
        :param decay_const: Decay constant to control the exponential decay of time values.
        :param downsample: Whether to downsample the time surface by half.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.decay_const = decay_const
        self.downsample = downsample

    def get_shape(self) -> Tuple[int, int, int]:
        if self.downsample:
            return (1, self.height // 2, self.width // 2)
        return (1, self.height, self.width)

    def create_surface_tensor(self, x: th.Tensor, y: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        Constructs a time surface using Torch tensors.
        :param x: x-coordinates of events.
        :param y: y-coordinates of events.
        :param time: timestamps of events.
        :return: Time surface as a Torch tensor.
        """
        device = x.device
        surface = th.full((self.height, self.width), fill_value=-1, dtype=th.float32, device=device)

        # Clip coordinates to ensure they are within bounds
        x = th.clamp(x, 0, self.width - 1)
        y = th.clamp(y, 0, self.height - 1)

        # Update the time surface with the latest event times
        for i in range(len(time)):
            xi, yi, ti = x[i], y[i], time[i]
            surface[yi, xi] = ti

        # Apply exponential decay
        max_time = time[-1]  # Latest event time
        surface = th.exp(-(max_time - surface) / self.decay_const)
        surface[surface < 0] = 0  # Ignore invalid values

        # Downsample the time surface if required
        if self.downsample:
            surface = th.nn.functional.interpolate(
                surface.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=(self.height // 2, self.width // 2),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)

        return surface.unsqueeze(0)  # Add channel dimension

    def create_surface_numpy(self, x: np.ndarray, y: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Constructs a time surface using NumPy arrays.
        :param x: x-coordinates of events.
        :param y: y-coordinates of events.
        :param time: timestamps of events.
        :return: Time surface as a NumPy array.
        """
        surface = np.full((self.height, self.width), fill_value=-1, dtype=np.float32)

        # Clip coordinates to ensure they are within bounds
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Update the time surface with the latest event times
        for i in range(len(time)):
            xi, yi, ti = x[i], y[i], time[i]
            surface[yi, xi] = ti

        # Apply exponential decay
        max_time = time[-1]  # Latest event time
        surface = np.exp(-(max_time - surface) / self.decay_const)
        surface[surface < 0] = 0  # Ignore invalid values

        # Downsample the time surface if required
        if self.downsample:
            surface = cv2.resize(surface, (self.width // 2, self.height // 2), interpolation=cv2.INTER_LINEAR)

        return surface[np.newaxis, :]  # Add channel dimension

    def construct(self,
                  x: Union[th.Tensor, np.ndarray],
                  y: Union[th.Tensor, np.ndarray],
                  pol: Union[th.Tensor, np.ndarray],  # Not used for time surface
                  time: Union[th.Tensor, np.ndarray]) -> Union[th.Tensor, np.ndarray]:
        if isinstance(x, th.Tensor):
            return self.create_surface_tensor(x, y, time)
        elif isinstance(x, np.ndarray):
            return self.create_surface_numpy(x, y, time)
        else:
            raise ValueError("Unsupported type for input data.")
