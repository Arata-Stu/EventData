import sys
sys.path.append("../")

import numpy as np
import torch as th
import argparse
from pathlib import Path
import cv2  # 画像保存用
from data.utils.h5reader import H5Reader
from data.utils.representation import EventFrame
from utils.timers import Timer

def main(h5_file_path, width, height, delta_t_ms, duration_t_ms):
    # delta_t と duration_t の単位をマイクロ秒に変換（タイムスタンプの単位に合わせる）
    delta_t_us = delta_t_ms * 1000
    duration_t_us = duration_t_ms * 1000
    h5_file = Path(h5_file_path)
    
    with H5Reader(h5_file, width=width, height=height) as reader:
        # ファイル内に設定された高さと幅を取得
        height, width = reader.get_height_and_width()
        # EventFrame のインスタンス生成
        event_frame = EventFrame(height=height, width=width, downsample=False)
        
        # 補正済みのタイムスタンプ配列を取得
        # タイムスタンプを明示的に符号付きの int64 に変換
        t_array = reader.time.astype(np.int64)
        start_time = int(t_array[0])
        end_time = int(t_array[-1])
        current_time = start_time

        # 動画保存のための VideoWriter の設定
        # fps は delta_t_ms から算出（例：100msステップなら fps=10）
        fps = 1000 / delta_t_ms
        output_video_filename = "event_video_3.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 動画のフレームサイズは (幅, 高さ)
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(output_video_filename, fourcc, fps, frame_size)

        frame_index = 0  # フレーム番号の管理
        while current_time <= end_time:
            # 現在時刻から duration_t_us だけ戻った時間範囲をウィンドウとして設定
            window_start_time = current_time - duration_t_us
            if window_start_time < start_time:
                window_start_time = start_time

            # np.searchsorted でウィンドウ内のイベントインデックスを取得
            idx_start = np.searchsorted(t_array, window_start_time, side='left')
            idx_end = np.searchsorted(t_array, current_time, side='right')

            # 指定範囲にイベントが存在しない場合はスキップ
            if idx_end <= idx_start:
                print(f"Skipping frame {frame_index}: no events in window [{window_start_time}, {current_time}]")
                current_time += delta_t_us
                frame_index += 1
                continue

            # 指定範囲のイベントデータを取得（NumPy 配列として）
            events = reader.get_event_slice(idx_start, idx_end, convert_2_torch=False)
            x = events['x']
            y = events['y']
            pol = events['p']
            t_vals = events['t']

            # Timer を利用して画像生成処理の時間を計測（任意）
            with Timer(timer_name=f"Event Frame"):
                frame_img = event_frame.construct(x, y, pol, t_vals)
            if isinstance(frame_img, th.Tensor):
                # PyTorch テンソルを NumPy 配列に変換
                frame_img = frame_img.detach().cpu().numpy()

            # frame_img は CHW 形式（チャネル, 高さ, 幅）のため、HWC 形式に変換
            frame_img_hwc = np.transpose(frame_img, (1, 2, 0))
            # OpenCV 用に RGB から BGR に変換
            frame_img_bgr = cv2.cvtColor(frame_img_hwc, cv2.COLOR_RGB2BGR)
            
            # 動画にフレームを書き込む
            video_writer.write(frame_img_bgr)
            
            print(f"フレーム {frame_index}: 時間窓 [{window_start_time}, {current_time}], イベント数 {len(t_vals)} を動画に追加")
            
            # 次のフレームへ進む
            frame_index += 1
            current_time += delta_t_us

        # VideoWriter を解放して動画ファイルを保存
        video_writer.release()
        print(f"動画 {output_video_filename} を保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 events to video.")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--width", type=int, required=True, help="Camera Width.")
    parser.add_argument("--height", type=int, required=True, help="Camera Height.")
    parser.add_argument("--delta_t_ms", type=int, required=True, default=100, help="Time interval in ms for stepping.")
    parser.add_argument("--duration_t_ms", type=int, required=True, help="Accumulation time in ms for event slices.")
    args = parser.parse_args()

    main(args.h5_file, args.width, args.height, args.delta_t_ms, args.duration_t_ms)
