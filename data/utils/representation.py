from typing import Tuple, Union, Optional

import numpy as np
import torch as th
import cv2


from abc import ABC, abstractmethod

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
        # 2チャンネルのカウント用テンソルを初期化 (0: ON, 1: OFF)
        counts = th.zeros((2, self.height, self.width), dtype=th.int32, device=device)
        
        # 座標をフレームサイズ内にクリップ
        x_clipped = th.clamp(x, min=0, max=self.width - 1)
        y_clipped = th.clamp(y, min=0, max=self.height - 1)
        
        # ON イベントのカウント
        on_mask = (pol == 1)
        if on_mask.sum() > 0:
            counts.index_put_((0, y_clipped[on_mask], x_clipped[on_mask]),
                              th.ones_like(x_clipped[on_mask], dtype=counts.dtype, device=device),
                              accumulate=True)
        
        # OFF イベントのカウント
        off_mask = (pol == 0)
        if off_mask.sum() > 0:
            counts.index_put_((1, y_clipped[off_mask], x_clipped[off_mask]),
                              th.ones_like(x_clipped[off_mask], dtype=counts.dtype, device=device),
                              accumulate=True)
        
        # 多数決: 差分 = ON - OFF
        diff = counts[0] - counts[1]
        
        # 出力画像を灰色（127）で初期化
        img = th.full((3, self.height, self.width), fill_value=127, dtype=th.uint8, device=device)
        
        # ON が多数の場合は白（255）、OFF が多数の場合は黒（0）に設定
        white_mask = (diff > 0)
        black_mask = (diff < 0)
        for c in range(3):
            img[c][white_mask] = 255
            img[c][black_mask] = 0
        
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
        Majority vote によるイベントフレームを作成する (NumPy版)。
        各ピクセルで ON と OFF のイベント数をカウントし、多数決を行う。
        :param x: x 座標
        :param y: y 座標
        :param pol: イベントの極性 (+1 for ON, 0 for OFF)
        :param time: タイムスタンプ（未使用）
        :return: RGB画像 (numpy.ndarray)
        """
        # 2チャンネルのカウント用配列を初期化 (0: ON, 1: OFF)
        counts = np.zeros((2, self.height, self.width), dtype=np.int32)
        
        # 座標をフレームサイズ内にクリップ
        x_clipped = np.clip(x, 0, self.width - 1)
        y_clipped = np.clip(y, 0, self.height - 1)
        
        # ON イベントのカウント
        on_mask = (pol == 1)
        if np.any(on_mask):
            np.add.at(counts[0], (y_clipped[on_mask], x_clipped[on_mask]), 1)
        
        # OFF イベントのカウント
        off_mask = (pol == 0)
        if np.any(off_mask):
            np.add.at(counts[1], (y_clipped[off_mask], x_clipped[off_mask]), 1)
        
        # 多数決: 差分 = ON - OFF
        diff = counts[0] - counts[1]
        
        # 出力画像を灰色（127）で初期化
        img = np.full((3, self.height, self.width), 127, dtype=np.uint8)
        
        # ON が多数の場合は白（255）、OFF が多数の場合は黒（0）に設定
        white_mask = diff > 0
        black_mask = diff < 0
        for c in range(3):
            img[c][white_mask] = 255
            img[c][black_mask] = 0
        
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
        """
        In case of fastmode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fastmode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
        """
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is None:
            self.count_cutoff = 255
        else:
            assert count_cutoff >= 1
            self.count_cutoff = min(count_cutoff, 255)
        self.fastmode = fastmode
        self.channels = 2
        self.downsample = downsample

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        if self.downsample:
            return 2 * self.bins, self.height // 2, self.width // 2
        return 2 * self.bins, self.height, self.width


    def create_frame_tensor(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
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
    
    def create_from_numpy(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        assert x.shape == y.shape == pol.shape == time.shape
        dtype = np.uint8 if self.fastmode else np.int16

        representation = np.zeros((self.channels, self.bins, self.height, self.width), dtype=dtype)
        if x.size == 0:
            representation = representation.reshape((2 * self.bins, self.height, self.width))
            return representation.astype(np.uint8)

        t0_int = time[0]
        t1_int = time[-1]
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_idx = np.clip((t_norm * self.bins).astype(np.int32), 0, self.bins - 1)

        indices = x + self.width * y + self.height * self.width * t_idx + self.bins * self.height * self.width * pol
        values = np.ones_like(indices, dtype=dtype)
        np.add.at(representation.flatten(), indices, values)
        representation = np.clip(representation, 0, self.count_cutoff)

        if self.downsample:
            new_height = self.height // 2
            new_width = self.width // 2
            representation_resized = np.zeros((self.channels, self.bins, new_height, new_width), dtype=np.uint8)
            for c in range(self.channels):
                for b in range(self.bins):
                    representation_resized[c, b] = cv2.resize(representation[c, b], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            representation = representation_resized

        return representation.reshape((2 * self.bins, self.height // 2, self.width // 2) if self.downsample else (2 * self.bins, self.height, self.width))


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
