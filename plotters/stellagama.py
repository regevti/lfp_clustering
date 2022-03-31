import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager, gridspec
from tqdm.auto import tqdm
import seaborn as sns
from pathlib import Path
import utils
from readers.mat_files import MatRecordingsParser
from detectors.shw import SharpWavesFinder
from scipy.optimize import curve_fit
from scipy.stats import linregress, ttest_ind
from scipy.io import savemat, loadmat

matplotlib.rcParams['pdf.fonttype'] = 42
# set paper font
font_dirs = ['/usr/share/fonts/truetype/myriad-pro-regular']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Myriad Pro'
animal_colors = {
    'SA03':	(0, 0.447, 0.741),
    'SA04': (0.850, 0.325, 0.098),
    'SA05': (0.929, 0.694, 0.125),
    'SA06': (0.494, 0.184, 0.556),
    'SA07': (0.466, 0.674, 0.188),
    'SA09': (0.301, 0.745, 0.933),
    'SA10': (0.635, 0.078, 0.184),
    'SA11': (0.741, 0.447, 0)
}
ALPHA = 0.6
cycles_compare = [
        ('SA07', 'SA07_SA09_23_06_21_Trial16_Trial30_18D'),
        ('SA07', 'SA07_SA09_17_06_21_Trial11_Trial25_27D'),
        ('SA07', 'SA07_AS09_19_06_21_Trial13_Trial27_35D')
]


class StellaPlotter:
    def __init__(self, output_folder='../output'):
        self.cycle_t = None  # time vector for cycles analyses, needed for plots
        self.cycles_rates = {}
        self.all_nights = {}
        self.temps = {}
        self.sw_dfs = {}
        self.avg_shw_shapes = {}
        self.summary_df = pd.DataFrame()
        self.bad_recordings = []
        self.tmpl = self.get_match_filter()
        assert Path(output_folder).exists() and Path(output_folder).is_dir()
        self.output_folder = Path(output_folder) / 'stella_figures'
        self.output_folder.mkdir(exist_ok=True)
        self.plots_folder.mkdir(exist_ok=True)
        self.load_cache()
        self.colormap = cm.get_cmap('coolwarm', len(set(self.temps.values())))
        self.colors = {int(k): self.colormap(i) for i, k in enumerate(sorted(set(self.temps.values())))}
        self.example_detector = None

    def analyze(self, mats_paths):
        for p in tqdm(mats_paths):
            try:
                i = p.name.find('SA')
                animal_id = p.name[i:i + 4]
                rec_id = -4 if p.parts[-3].startswith('Record') else -3
                rec_name = p.parts[rec_id]
                name = (animal_id, rec_name)
                rp, swf = self.init_parsers(p, animal_id)
                r, self.cycle_t, _ = swf.calc_cycle_sw_rate(wt=20, group_length=150)
                rf = swf.calc_sw_rate(60 * 60, overlap=0.75, label=None, lights_off_only=True)
                self.cycles_rates[name] = r
                self.all_nights[name] = rf
                self.temps[name] = rp.temp
                self.sw_dfs[name] = swf.shw_df.copy()
                if name in cycles_compare:
                    self.calc_average_shw_shape(rp, name)
            except Exception as exc:
                print(f'ERROR; {exc}; {p}')
        self.temps = {k: v for k, v in sorted(self.temps.items(), key=lambda item: item[1])}
        self.analysis_summary()
        self.save_cache()

    def analysis_summary(self):
        for i, (name, temp) in enumerate(self.temps.items()):
            rate = self.cycles_rates[name]
            sw_df = self.sw_dfs[name]
            all_night = self.all_nights[name]
            q = all_night.datetime.dt.strftime('%H:%M:%S')
            all_night = all_night[(q >= '19:00') | (q <= '07:00')]
            self.summary_df.loc[i, 'animal_id'] = name[0]
            self.summary_df.loc[i, 'name'] = name[1]
            self.summary_df.loc[i, 'temp'] = temp
            # auto-correlation
            self.summary_df.loc[i, ['t_cyc', 'power', 'power_norm', 'r', 'lag']] = utils.autocorr(rate, self.cycle_t)
            # ShW widths
            self.summary_df.loc[i, 'width'] = sw_df.query('power>0.4').width.mean()
            self.summary_df.loc[i, 'depth'] = sw_df.query('power>0.4').depth.mean()
            # total ShW
            self.summary_df.loc[i, 'total_sw'] = len(sw_df)
            # ShW/cycle
            self.summary_df.loc[i, 'sw_per_cyc'] = sw_df.groupby('signal').start.count().mean()
            # max cycle rate value
            self.summary_df.loc[i, 'max_cyc_rate'] = rate[:len(rate) // 2].max()
            self.summary_df.loc[i, 'p2v'] = rate[:len(rate) // 2].max() - rate[len(rate) // 2:].min()
            self.summary_df.loc[i, 'all_night_avg'] = all_night.rate.mean()
        self.bad_recordings = [tuple(x) for x in self.summary_df.query('power<0.3')[['animal_id',
                                                                                     'name']].to_records(index=False)]

    def calc_average_shw_shape(self, rp, name):
        v_, _ = rp.read()
        sf = self.sw_dfs[name].copy()
        sf.group.fillna(0, inplace=True)
        self.avg_shw_shapes[name] = []
        for i, row in sf.iterrows():
            self.avg_shw_shapes[name].append(v_[int(row.start):int(row.end)])
        self.avg_shw_shapes[name] = np.vstack(self.avg_shw_shapes[name]).mean(axis=0)

    def temperature_figure(self):
        """Plot all figures about ShW vs. temperature"""
        fig, axes = plt.subplots(2, 5, dpi=130, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 12, 12, 12, 12]})
        self.plot_sw_around_cycle_transition(axes[0, 1])
        self.plot_rate_all_night(axes[0, 2])
        self.plot_sw_shapes(axes[1, 1])

        for ax in axes[:, -1]:  # remove the underlying axes
            ax.remove()
        gs = axes[0, 3].get_gridspec()
        ax_big = fig.add_subplot(gs[:, -1])
        h, l = self.plot_sw_cycle_oscil_freq(ax_big)

        l, h = zip(*sorted(zip(l, h), key=lambda t: t[0]))
        for lh in h:
            lh.set_alpha(ALPHA)
        axes[1, 0].legend(h, l, borderaxespad=0, fontsize=8, bbox_to_anchor=(1, 1))
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        self.plot_sw_widths(axes[1, 2])
        self.plot_sw_depths(axes[1, 3])
        self.plot_sw_avg_rate(axes[0, 3])
        fig.tight_layout()

        min_temp, max_temp = int(min(self.temps.values())), int(max(self.temps.values()))
        sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=plt.Normalize(vmin=min_temp, vmax=max_temp))
        cbar_ax = fig.add_axes([0.05, 0.78, 0.01, 0.17])
        fig.colorbar(sm, cax=cbar_ax, ticks=[min_temp, max_temp])

    def supplementary_figure(self):
        swf = self.load_example_detector()
        fig = plt.figure(tight_layout=True, figsize=(10, 5), dpi=130)
        cols = 5
        gs = gridspec.GridSpec(3, cols, height_ratios=[1, 3, 3], width_ratios=[1, 1, 1, 1, 6])
        ax = fig.add_subplot(gs[0, 1])
        c = 'darkblue'
        ax.plot(self.tmpl, c, linewidth=2)
        ax.arrow(len(self.tmpl), self.tmpl.mean(), len(self.tmpl) / 2, 0, color=c, head_width=20)
        ax.arrow(0, self.tmpl.mean(), -len(self.tmpl) / 2, 0, color=c, head_width=20)
        ax.axis('off')
        swf.plot_sharp_waves_detection(26894, 26907, fig.add_subplot(gs[1, :cols - 1]),
                                        fig.add_subplot(gs[2, :cols - 1]))
        ax = fig.add_subplot(gs[1, cols - 1])
        self.plot_rate_all_night(ax, only_mid=True, is_legend=True)

    def phase_figure(self):
        gdf = []
        swf = self.load_example_detector()
        for name, sw_df in self.sw_dfs.items():
            if name in self.bad_recordings:
                continue
            group_rate = sw_df.groupby('group').signal.count()
            group_rate /= group_rate.sum()
            group_rate = group_rate.to_dict()
            group_rate.update({'animal_id': name[0], 'name': name[1]})
            gdf.append(group_rate)
        gdf = pd.DataFrame(gdf).rename(columns={1.0: 'SWS', 2.0: 'REM'})

        phase_db = loadmat('../output/db_phase.mat')['pdb'].flatten()
        phase_db = (phase_db - phase_db.min()) / (phase_db.max() - phase_db.min())

        fig, axes = plt.subplots(1, 2, dpi=130, figsize=(6, 3))
        t_phase, r_phase = swf.plot_phase_shw_cycle(ax=axes[0])
        axes[0].plot(t_phase, phase_db, label=r'$\delta/\beta$')
        axes[0].legend()
        sns.violinplot(data=gdf, ax=axes[1])
        axes[1].set_ylabel('In-cycle ShW ratio')
        tt, p_val = ttest_ind(gdf.SWS, gdf.REM, equal_var=False)
        print(f't={tt:.3f} , P={p_val:.1e}')
        fig.tight_layout()

    def load_example_detector(self) -> SharpWavesFinder:
        if self.example_detector is not None:
            return self.example_detector
        p = Path(
            '/media/sil2/Data/Lizard/Stellagama/SA09_SA07/SA07_SA09_17_06_21_Trial11_Trial25_27D/Record Node 117/regev_cache/decimate_rec_SA07.mat')
        _, swf = self.init_parsers(p, 'SA07')
        return swf

    def init_parsers(self, p, animal_id):
        rp = MatRecordingsParser(p.parent.parent.as_posix(), channel=None, is_debug=False, animal_id=animal_id,
                                 mat_only=True, window=600, overlap=0.75, wavelet=None, lowpass=40)
        swf = SharpWavesFinder(rp, shw_duration=1.2, is_debug=False)
        swf.train(mfilt=self.tmpl[::-1], thresh=0.25)
        return rp, swf

    def plot_sw_around_cycle_transition(self, ax=None):
        if ax is None:
            ax = plt.subplot(figsize=(6, 3), dpi=140)
        t_ = self.cycle_t[2:-2] - self.cycle_t[-1] / 2
        for name, temp in self.temps.items():
            if name not in cycles_compare:
                continue
            rate = self.cycles_rates[name]
            ax.plot(t_, rate[2:-2] / 20, color=self.colors[int(temp)])
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Time [sec]', fontsize=9)
        ax.set_ylabel('Rate [#ShW/sec]', fontsize=9)

    def plot_rate_all_night(self, ax, only_mid=False, is_legend=None):
        if ax is None:
            ax = plt.subplot(figsize=(6, 3), dpi=140)
        for animal_id in ['SA07']:
            averaged = []
            n_recs = 0
            for name, temp in self.temps.items():
                if name in self.bad_recordings or name[0] != animal_id:
                    continue
                rf = self.all_nights[name].copy().reset_index()[['rate', 'time', 'datetime']]
                rf.time = rf.time / 3600
                rf.datetime = rf.datetime.dt.strftime('%H:%M')
                rf_vec = rf.set_index('datetime')['rate']
                averaged.append(rf_vec.copy())
                n_recs += 1
                if name in cycles_compare:
                    if only_mid and name != ('SA07', 'SA07_SA09_17_06_21_Trial11_Trial25_27D'):
                        continue
                    sns.lineplot(data=rf, x='datetime', y='rate', color=self.colors[int(temp)], ax=ax,
                                 label=is_legend and f'{int(temp)}ºC', zorder=10)
            if n_recs == 0:
                continue
            avf = pd.DataFrame(averaged).transpose()
            sns.lineplot(x=avf.index.values, y=avf.mean(axis=1), color='k', ax=ax, label=is_legend and 'Average',
                         ci=None)
            ax.fill_between(avf.index.values, avf.mean(axis=1) - avf.sem(axis=1), avf.mean(axis=1) + avf.sem(axis=1),
                            alpha=0.5)
        ax.set_xlabel('Hour', fontsize=9)
        ax.set_ylabel('Rate [#ShW/hour]', fontsize=9)
        ax.set_xticks(['19:00', '00:00', '07:00'])
        ax.axvspan('19:00', '07:00', facecolor='silver', alpha=0.3)
        if is_legend:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, l)
        return averaged

    def plot_sw_shapes(self, ax):
        t_ = np.arange(-0.4, 0.4, 1 / 400)
        for name, v in self.avg_shw_shapes.items():
            temp = self.temps[name]
            ax.plot(t_, v, label=f'{temp}', color=self.colors[int(temp)])
        ax.set_xlabel('Time [sec]', fontsize=9)
        ax.set_ylabel('Voltage [mV]', fontsize=9)
        ax.set_xticks([-0.5, 0, 0.5], ['-0.5', '0', '0.5'])

    def plot_sw_cycle_oscil_freq(self, ax, T0=17, p0=(2.2, 1.7)):
        data = self.summary_df.query('power>0.3').copy()
        data['freq'] = 1 / (data.t_cyc / 60)
        xdata = data['temp'].values
        ydata = data['freq'].values
        def q10(T, *P): return P[1] * P[0] ** ((T - T0) / 10)
        popt, pcov = curve_fit(q10, xdata, ydata, p0=p0)
        sns.scatterplot(data=data, x='temp', y='freq', hue='animal_id', ax=ax, palette=animal_colors, alpha=ALPHA,
                        edgecolor='black', s=18)
        ax.plot(xdata, q10(xdata, *popt), 'k-', label=r'$Q_{10}$' + f' = {popt[0]:.2f}')
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[-1:], l[-1:])
        ax.set_xlabel('Temperature [ºC]', fontsize=9)
        ax.set_ylabel('Oscil. freq [1/min]', fontsize=9)
        ax.set_xticks([20, 25, 30, 35])
        return h[:-1], l[:-1]

    def _plot_general_metric(self, ax, col, use_stat=True, label=None):
        label = label or col
        assert col in self.summary_df.columns, f'Column {col} is unknown'
        data = self.summary_df.query('power>0.3 and animal_id not in ["SA10","SA11"]').copy()
        # group normalization
        for animal_id, idx in data.groupby('animal_id').groups.items():
            data.loc[idx, col] = (data.loc[idx, col] - data.loc[idx, col].mean()) / data.loc[idx, col].std()
        sns.scatterplot(data=data, x='temp', y=col, hue='animal_id', ax=ax, palette=animal_colors, alpha=ALPHA,
                        edgecolor='black', s=18)
        if use_stat:
            slope, intercept, r, p_r, se = linregress(data.temp.values, data[col].values)
            c, p_c = utils.corrcoef(np.vstack([data.temp.values, data[col].values]))
            print(f'Statistics for {col} vs. temp: r^2={r ** 2:.3f},p-value={p_r:.0e}, '
                  f'corr={c[0, 1]:.3f},p-value={p_c[0, 1]:.0e}')
            X = data.temp.values.reshape(-1, 1)
            ax.plot(X.squeeze(), X.squeeze() * slope + intercept, 'k')
        ax.legend([], [], frameon=False)
        ax.set_xlabel('Temperature [ºC]', fontsize=9)
        ax.set_ylabel(f'norm. ShW {label}', fontsize=9)
        ax.legend([], [], frameon=False)
        ax.set_xticks([20, 25, 30, 35])

    def plot_sw_depths(self, ax):
        self._plot_general_metric(ax, 'depth')

    def plot_sw_widths(self, ax):
        self._plot_general_metric(ax, 'width')

    def plot_sw_avg_rate(self, ax):
        self._plot_general_metric(ax, 'all_night_avg', use_stat=False, label='mean rate')

    @staticmethod
    def get_match_filter():
        with open('../output/template_filter.np', 'rb') as f:
            tmpl = np.load(f)
        return tmpl

    @property
    def cached_variables(self):
        return ['cycle_t', 'cycles_rates', 'all_nights', 'temps', 'sw_dfs', 'avg_shw_shapes']

    def load_cache(self):
        assert self.cache_filename.exists(), f'{self.cache_filename} does not exist'
        with self.cache_filename.open('rb') as f:
            d = pickle.load(f)
        for k in self.cached_variables:
            assert k in d.keys(), f'{k} is missing in the cache file from {self.cache_filename}'
            setattr(self, k, d[k])
        self.analysis_summary()

    def save_cache(self):
        d = {k: getattr(self, k) for k in self.cached_variables}
        with self.cache_filename.open('wb') as f:
            pickle.dump(d, f)

    @property
    def plots_folder(self):
        return self.output_folder / 'plots'

    @property
    def cache_filename(self):
        return self.output_folder / 'cache.pkl'

