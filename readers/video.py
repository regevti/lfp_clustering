from typing import Union
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from ctypes import c_int32, c_bool
import time
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

run_counter = mp.Value(c_int32)
run_counter_lock = mp.Lock()
extracted_frames_counter = mp.Value(c_int32)
extracted_frames_counter_lock = mp.Lock()
max_frames_reached = mp.Value(c_bool, False, lock=False)


class VideoFeatureExtraction:
    def __init__(self, video_file=None):
        self.video_file = video_file
        self.frames = None
        self.frames_ids = None
        self.features = None

    def run(self, min_frames_dist=2, max_frames=5e4, n_cpu=20, is_save=True):
        assert Path(self.video_file).exists(), 'video file does not exist'
        vr = VideoReader(self.video_file, min_dist=min_frames_dist, max_frames=max_frames)
        self.frames, self.frames_ids = vr.multiprocess_load(n_cpu=n_cpu)
        self.features = ResNetPretrained().feature_extraction(self.frames)
        if is_save:
            self.save_cache()

    def save_cache(self, save_path=None):
        save_path = save_path or self.cache_dir / f'{datetime.now().strftime("%Y%m%dT%H%M%S")}.h5'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset("features", data=self.features)
            f.create_dataset("frames", data=self.frames)
            f.create_dataset("frames_ids", data=self.frames_ids)

    def load_cache(self, save_path=None):
        if save_path is None:
            cache_files = list(self.cache_dir.glob('*.h5'))
            assert len(cache_files) > 0, f'No cache files found in {self.cache_dir}'
            save_path = cache_files[-1]
        with h5py.File(save_path, 'r') as f:
            self.features = np.array(f['features'])
            self.frames, self.frames_ids = np.array(f['frames']), np.array(f['frames_ids'])
        return self

    @property
    def cache_dir(self):
        return Path(self.video_file).parent.parent / 'regev_cache' / 'resnet_features'


class VideoReader:
    def __init__(self, video_path, min_dist=None, start_frame=0, stop_frame=None, max_frames=None):
        self.video_path = video_path
        self.min_dist = min_dist
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.last_frame = None

        self.vid = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stop_frame = stop_frame or self.n_frames - 1
        assert isinstance(start_frame, int) and 0 <= start_frame < self.n_frames, \
            f'bad start frame: {start_frame}; total frames={self.n_frames}; type={type(start_frame)}'
        assert isinstance(self.stop_frame, int) and 0 <= self.stop_frame < self.n_frames, f'bad stop frame'
        if start_frame:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current_frame_id = start_frame

    def run(self):
        pass

    def read_frame(self, pbar=None):
        while self.start_frame <= self.current_frame_id <= self.stop_frame and not max_frames_reached.value:
            ret, frame = self.vid.read()
            if self.last_frame is None:
                self.last_frame = frame
                return frame
            self.current_frame_id += 1
            with run_counter_lock:
                run_counter.value += 1
            if pbar is not None:
                pbar.update(1)
            if ret:
                if self.min_dist and self.image_dist(frame, self.last_frame) < self.min_dist:
                    continue
                else:
                    self.last_frame = frame
                    return frame

    def resnet_embedding(self, max_features=None):
        rsnt = ResNetPretrained()
        frame, features, frame_ids = True, [], []
        with tqdm(total=self.n_frames) as pbar:
            while frame is not None:
                frame = self.read_frame(pbar)
                if frame is None:
                    break
                _, x = rsnt(frame)
                features.append(x.detach().cpu().numpy())
                frame_ids.append(self.current_frame_id)
                if max_features and len(features) >= max_features:
                    pbar.update(self.n_frames)
                    break
                pbar.set_description(f'(#Features={len(features)}) | Frames')
        features, frame_ids = np.vstack(features), np.array(frame_ids)
        print(f'Finish embedding. Feature shape: {features.shape}')
        return features, frame_ids

    @staticmethod
    def transform_image(image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image)).convert('RGB')
        image = transforms.Resize((224, 224))(image)
        return np.array(image)

    def load(self):
        frames, frame_ids = [], []
        while not max_frames_reached.value:
            frame = self.read_frame()
            if frame is None:
                break
            with extracted_frames_counter_lock:
                if not max_frames_reached.value:
                    frame = self.transform_image(frame)
                    frames.append(frame)
                    extracted_frames_counter.value += 1
                    frame_ids.append(self.current_frame_id)
                if self.max_frames and extracted_frames_counter.value >= self.max_frames:
                    max_frames_reached.value = True

        frame_ids = np.array(frame_ids)
        frames = np.stack(frames) if frames else np.array([])
        return frames, frame_ids

    def multiprocess_load(self, n_cpu=10):
        assert n_cpu <= mp.cpu_count(), f'requested CPUs are higher than system cpu count: {mp.cpu_count()}'
        print(f'Start multiprocess load with {n_cpu} cores, on {self.n_frames} frames')
        result = None
        run_counter.value, extracted_frames_counter.value, max_frames_reached.value = 0, 0, False
        with tqdm(total=self.n_frames) as pbar:
            with mp.Pool(processes=n_cpu) as pool:
                try:
                    future = pool.map_async(_multiprocessing_video_reader,
                                            ((self.video_path, self.min_dist, int(g[0]), int(g[-1]), self.max_frames)
                                             for g in np.array_split(np.arange(self.n_frames), n_cpu)))
                    while not future.ready():
                        update_progress_bar(pbar)
                        time.sleep(1)
                    result = future.get()
                except Exception as exc:
                    print(f'ERROR: {exc}')
                finally:
                    pool.close()
                    update_progress_bar(pbar)

        frames = np.vstack([r[0] for r in result if len(r[0]) > 0])
        frame_ids = np.hstack([r[1] for r in result if len(r[1]) > 0])
        return frames, frame_ids

    @staticmethod
    def image_dist(img1, img2):
        return cv2.absdiff(img1, img2).mean()


class ResNetPretrained(nn.Module):
    def __init__(self, rescale_size=(224, 224)):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.transformer(x).unsqueeze(0).cuda()
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

    def feature_extraction(self, frames: np.ndarray):
        """
        Extract features using resnet
        @param frames: [n_frames, 3, 224, 224]
        @return: embedded features [n_frames, 2040]
        """
        features = []
        for frame in tqdm(frames):
            _, x = self(frame)
            features.append(x.detach().cpu().numpy())
        return np.vstack(features)


def update_progress_bar(pbar):
    """async update of the tqdm progress bar according to the multiprocess counters"""
    if run_counter.value != 0:
        with run_counter_lock:
            increment = run_counter.value
            run_counter.value = 0
            pbar.update(n=increment)

    with extracted_frames_counter_lock:
        pbar.set_description(f'(#Extracted Frames={extracted_frames_counter.value}) | Frames')


def _multiprocessing_video_reader(args):
    # args: (video_path, min_dist, start_frame, stop_frame, min_dist)
    vr = VideoReader(*args)
    return vr.load()
