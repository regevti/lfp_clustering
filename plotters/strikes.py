import json
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from readers.base import NeoReader


# duration before and after the strike time
SEC_BEFORE = 4
SEC_AFTER = 4


class StrikesOE:
    def __init__(self, rp: NeoReader):
        self.rp = rp
        self.session_dir = rp.root_dir.parent.parent
        self.behavior_events = None
        self.bf = self.load_strikes_data()
        self.miny = 0
        self.segs = []
        self.t = np.arange(-SEC_BEFORE, SEC_AFTER, 1 / self.rp.fs)

    def load_strikes_data(self) -> pd.DataFrame:
        bf = self.load_strikes_from_events()
        trig = self.load_triggers()
        frames_ts, frames_dt = self.load_frames_times()
        oe_frames_ts = self.get_frames_in_oe_time(trig, frames_ts)
        bf = self.convert_reptilearn_events_time(bf, frames_dt, oe_frames_ts)
        self.load_strikes_frames_events(oe_frames_ts)
        return bf

    def load_strikes_from_events(self):
        paths = list(self.session_dir.rglob('events.csv'))
        assert len(paths) == 1
        bf = pd.read_csv(paths[0]).query('event=="screen_touch"').drop(columns=['event'])
        event_time_col = bf.rename(columns={'time': 'event_time'})['event_time']
        bf = bf.value.apply(lambda x: pd.Series(json.loads(x)))
        bf = pd.concat([event_time_col, bf], axis=1)
        bf['time'] = pd.to_datetime(bf.time, unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
        return bf

    def load_signals(self):
        self.segs = []
        for i, row in tqdm(self.bf.iterrows(), total=len(self.bf)):
            t_strike = row.oe_time
            v_, t_ = self.rp.read(t_start=t_strike - SEC_BEFORE, t_stop=t_strike + SEC_AFTER)
            v_ = np.interp(self.t, t_ - t_[0] - SEC_BEFORE, v_)
            if v_.min() < self.miny:
                self.miny = v_.min()
            self.segs.append(v_.flatten())

    def plot_signals(self):
        fig, axes = plt.subplots(2, 1, figsize=(25, 10))
        for v_ in self.segs:
            axes[0].plot(self.t, v_)
        axes[1].plot(self.t, np.vstack(self.segs).mean(axis=0))
        for ax in axes:
            ax.axvline(0, color='k')
        axes[0].set_title(str(self.rp))

    def convert_reptilearn_events_time(self, bf, frames_dt, oe_frames_ts):
        def _convert(t: pd.Timestamp):
            assert frames_dt[0] <= t <= frames_dt[-1], f'{t} is not in range: [{frames_dt[0]}, {frames_dt[-1]}]'
            i = np.argmin(np.abs((frames_dt - t).total_seconds()))
            return pd.Series([oe_frames_ts[i], i])

        bf[['oe_time', 'frame_id']] = bf.time.apply(_convert)
        bf['frame_id'] = bf.frame_id.astype(int)
        return bf

    def load_triggers(self):
        trig = self.rp.reader.get_event_timestamps(event_channel_index=0)[0]
        trig = self.rp.reader.rescale_event_timestamp(trig)
        trig = trig - self.rp.reader.segment_t_start(block_index=0, seg_index=0)
        return trig[::2]

    def load_frames_times(self):
        video_files = list(self.session_dir.rglob('top*.mp4'))
        assert len(video_files) == 1
        frames_ts = pd.read_csv(video_files[0].with_suffix('.csv')).timestamp.values
        frames_dt = pd.to_datetime(frames_ts, unit='s').tz_localize("utc").tz_convert('Asia/Jerusalem')
        return frames_ts, frames_dt

    def load_strikes_frames_events(self, oe_frames_ts):
        if not self.behavior_events_file_path.exists():
            print(f'file {self.behavior_events_file_path} does not exist')
            return
        self.behavior_events = pd.read_csv(self.behavior_events_file_path)
        for c in self.behavior_events.columns:
            self.behavior_events[f'{c}_time'] = self.behavior_events[c].map(
                lambda x: oe_frames_ts[int(x)] if x and not np.isnan(x) else None)

    def plot_behavior_events(self):
        if self.behavior_events is None:
            print('unable to plot; behavior events does not exist')
            return

        idx = self.behavior_events[self.behavior_events.isna().any(axis=1)].index
        bhf = self.behavior_events.drop(idx)

        fig, ax = plt.subplots(1, 1, dpi=150)
        vs, ts = [], []
        for i, row in bhf.iterrows():
            v_, t_ = self.rp.read(t_start=row.approach_time - 1, t_stop=row.strike_time + 1)
            vs.append(v_)
            ts.append(t_ - t_[0])

        max_index = int(np.argmin([t[-1] for t in ts]))
        global_t = ts[max_index]
        for i, (v_, t_) in enumerate(zip(vs, ts)):
            v_ = np.interp(global_t, t_, v_)
            ax.plot(global_t, v_ + (i * 300), color='k', linewidth=0.5)

        lines = {'approach': 'b', 'tongue': 'g', 'strike': 'r'}
        for l, color in lines.items():
            row = bhf.iloc[max_index]
            ax.axvline(row[f'{l}_time'] - row.approach_time + 1, color=color, linewidth=1.5, label=l)
        ax.legend()
        ax.set_yticks([])
        ax.set_xlabel('Time [sec]')

    def extract_relevant_frames_to_files(self, sec_back=3):
        video_files = list(self.session_dir.rglob('top*.mp4'))
        frames_ts = pd.read_csv(video_files[0].with_suffix('.csv')).timestamp.values
        fps = 1 / np.diff(frames_ts).mean()
        assert len(video_files) == 1
        n_frames_back = round(fps * sec_back)
        for frame_id in tqdm(self.bf.frame_id.values):
            frame_dir = self.cache_dir / str(frame_id)
            frame_dir.mkdir(exist_ok=True, parents=True)
            start_frame = frame_id - n_frames_back
            cap = cv2.VideoCapture(video_files[0].as_posix())
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in range(start_frame, frame_id + 1):
                ret, frame = cap.read()
                cv2.imwrite((frame_dir / f'{str(i)}.jpg').as_posix(), frame)
            cap.release()

    @staticmethod
    def get_frames_in_oe_time(oe_trig, frames_ts, mode='same'):
        i = np.where(np.diff(oe_trig) > 1)[0][0]
        frames_ts = frames_ts - frames_ts[0] + oe_trig[i+1]
        return frames_ts

    @property
    def cache_dir(self) -> Path:
        return self.session_dir / 'regev_cache'

    @property
    def behavior_events_file_path(self) -> Path:
        return self.cache_dir / 'strikes_events_in_frames.csv'
