import re
from datetime import datetime, time
import pandas as pd
import numpy as np
import xarray as xr
import h5py
from utils import butter_lowpass_filter
from .base import Reader


class MatRecordingsParser(Reader):
    def __init__(self, root_dir, channel, is_debug=True, animal_id=None, mat_only=False, lowpass=None):
        super().__init__(root_dir, channel, is_debug)
        self.xls_table = pd.read_excel('/media/sil2/Data/Lizard/Stellagama/brainStatesSS.xlsx')
        self.animal_id = animal_id
        self.mat_only = mat_only
        self.lowpass = lowpass
        self.temp = None
        self.t_start = None
        self.lights_off_id = None
        self.lights_on_id = None
        self.cache_dir_path.mkdir(exist_ok=True, parents=True)
        self.v, self.t = self.get_all_recording()
        self.fs = np.round(1 / np.mean(np.diff(self.t)))
        if self.lowpass is not None:
            self.v = butter_lowpass_filter(self.v.flatten(), lowpass, self.fs, order=5)
        self.sc = self.load_slow_cycles()

    def get_all_recording(self):
        if self.mat_only and not self.mat_path.exists():
            raise Exception(f'Unable to find {self.mat_path}')
        if self.mat_path.exists():
            self.print(f'Loading .mat file...')
            return self.load_mat()
        elif self.all_recording_path.exists():
            data = xr.open_dataset(self.all_recording_path)
            v = data.__xarray_dataarray_variable__.to_numpy()
            t = data.time.to_numpy()
            return v, t

    def load_mat(self):
        with h5py.File(self.mat_path, 'r') as f:
            v = np.array(f['v']).flatten()
            t = np.array(f['t']).flatten()
            if not self.channel:
                self.channel = int(np.array(f['channel']).flatten()[0])
            self.load_temp_from_mat(f)
            self.load_start_time_from_mat(f, t)
        return v, t

    def load_temp_from_mat(self, f):
        try:
            temp = np.array(f['temp']).flatten()[0]
            assert not np.isnan(temp)
        except Exception:
            m = re.search(r'(\d{2})D$', self.root_dir.parts[-2])
            if m:
                temp = m.group(1)
            else:
                m = self.xls_table.query(f'folder=="{self.root_dir}"')
                if m.empty:
                    raise Exception(f'Unable to find temperature for {self.root_dir}')
                else:
                    temp = m.Temp.values[0]
        self.temp = float(temp)

    def load_start_time_from_mat(self, f, t):
        try:
            sdate = np.array(f.get('start_date'), dtype='uint8').tobytes().decode('utf-8')
            assert not isinstance(sdate, str) or len(sdate) > 0
            self.t_start = pd.to_datetime(sdate)
            tv = pd.Series(self.t_start + t.astype('timedelta64[s]'))
            lights_off = datetime.combine(tv.dt.date[0], time(hour=19))
            lights_on = datetime.combine(tv.dt.date[tv.index[-1]], time(hour=7))
            self.lights_off_id = (tv - lights_off).abs().idxmin()
            self.lights_on_id = (tv - lights_on).abs().idxmin()
        except ImportError as exc:
            print(f'Unable to parse start_date; {exc}')

    @property
    def all_recording_path(self):
        return self.cache_dir_path / f'all_recordings_ch{self.channel}.nc'

    @property
    def mat_path(self):
        return self.cache_dir_path / f'decimate_rec_{self.animal_id}.mat'



    # self.print(f'Initializing an open-ephys reader...')
    # reader = OEReader(self.root_dir)
    # v, t = reader.get_data([self.channel], reader.start_time.base, reader.duration.base)
    # data = xr.DataArray(v.reshape((len(v), 1)), [t, [self.channel]], dims=['time', 'channel'])
    # data.to_netcdf(self.all_recording_path)
    # return v, t