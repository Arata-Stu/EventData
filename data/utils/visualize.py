import numpy as np
from einops import rearrange, reduce
import matplotlib.pyplot as plt

def ev_repr_to_img(input: np.ndarray):
    """
    イベント表現 (正の極性と負の極性) を RGB 画像に変換します。
    """
    ch, ht, wd = input.shape[-3:]
    assert ch > 1 and ch % 2 == 0, "Input channels must be a positive even number."
    ev_repr_reshaped = rearrange(input, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img

def visualize(input: np.ndarray, title: str = "Image Visualization"):
    """
    入力を可視化します。
    input: 入力テンソル (形状: [C, H, W])
    - チャンネルが2の倍数の場合: イベント表現とみなして可視化
    """
    ch = input.shape[-3]
    
    if ch % 2 == 0:
        # チャンネル数が2の倍数なら`ev_repr_to_img`を使用
        img = ev_repr_to_img(input)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        raise ValueError("Invalid input Channel")
