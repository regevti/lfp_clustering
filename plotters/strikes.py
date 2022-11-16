import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from readers.base import NeoReader


# duration before and after the strike time
SEC_BEFORE = 2
SEC_AFTER = 2


class StrikesOE:
    def __init__(self, rp: NeoReader):
        self.rp = rp
        self.session_dir = rp.root_dir.parent.parent
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
            return oe_frames_ts[i]

        bf['oe_time'] = bf.time.apply(_convert)
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

    @staticmethod
    def get_frames_in_oe_time(oe_trig, frames_ts, mode='same'):
        # oe_trig, frames_ts = oe_trig.copy(), frames_ts.copy()
        # x = (oe_trig - oe_trig.min()) / (oe_trig.max() - oe_trig.min())
        # y = (frames_ts - frames_ts.min()) / (frames_ts.max() - frames_ts.min())
        # corr = correlate(x, y, mode=mode)
        # lags = correlation_lags(x.size, y.size, mode=mode)
        # lag = lags[np.argmax(corr)]
        # print(f'Lag index: {lag}, {y.size}')
        # plt.figure()
        # plt.plot(lags, corr)
        # if lag > 0:
        #     oe_trig = oe_trig[lag:]
        # elif lag < 0:
        #     frames_ts = frames_ts[lag:]
        i = np.where(np.diff(oe_trig) > 1)[0][0]
        frames_ts = frames_ts - frames_ts[0] + oe_trig[i+1]
        return frames_ts
