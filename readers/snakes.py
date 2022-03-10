import xmltodict
from .base import Reader


class SnakeReader(Reader):
    def __init__(self, root_dir, channel):
        super().__init__(root_dir, channel)
        assert self.root_dir.exists() and self.root_dir.is_dir(), f'path {root_dir} not exists or not a folder'
        info = self.load_info()
        self.fs = float(info['Acquisition']['SamplingRate'])
        self.files = info['Acquisition']['Files']['File']
        self.channels = info['Acquisition']['Channels']['Channel']
        self.channel_names = [f['Name'] for f in self.channels]
        self.units = 'mV'

    def read(self, i_start=None, i_stop=None):
        assert isinstance(self.channel, int) and 1 <= self.channel <= len(self.files), 'Channel is out of bound'
        file_id = self.channel - 1
        bin_file = self.files[file_id]['FileName']
        start_time = self.files[file_id]['TStart']

    def load_info(self):
        assert 'Salazard B2.exp' in [p.name for p in self.root_dir.glob('*')]
        with open(f'{self.root_dir}/Salazard B2.exp', 'r') as f:
            xml = f.read()

        return xmltodict.parse(xml)['Animal']
