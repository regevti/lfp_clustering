from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches

from detectors.motives import MotifFinder, COLORS
from readers.video import VideoFeatureExtraction
import utils


class BrainBehavior:
    def __init__(self, recording_path=None, video_file=None):
        """
        Analysis class for brain and behavior
        @param recording_path: Path of recording folder or (.gz) cache file for the motives class
        @param video_file: The video file or (.h5) cache file
        """
        self.recording_path = Path(recording_path)
        self.video_file = Path(video_file)
        self.mf: MotifFinder = None
        self.vfe: VideoFeatureExtraction = None
        self.load()

    def load(self):
        if self.recording_path.parent.parent.name == 'motives':
            print(f'loading motives analysis cache from: {self.recording_path}')
            self.mf = MotifFinder(cache_dir=self.recording_path)
        else:
            assert self.recording_path.exists() and self.recording_path.is_dir(), 'bad recording path'

        print(self.video_file.suffix == '.h5')
        if self.video_file.suffix == '.h5':
            print(f'loading video features cache from: {self.video_file}')
            self.vfe = VideoFeatureExtraction().load_cache(self.video_file)
        else:
            assert self.video_file.exists(), 'video_file does not exist'

    def tsne_embedding(self, n_components=2):
        return TSNE(n_components=n_components, learning_rate='auto', init='random').fit_transform(self.vfe.features)

    def cluster_2d(self, distance_threshold=0.2, is_plot=False):
        X_embedded = self.tsne_embedding()
        X_normalized = StandardScaler().fit_transform(X_embedded)
        clustering = AgglomerativeClustering(distance_threshold=distance_threshold,
                                             n_clusters=None, linkage="single").fit(X_normalized)
        labels = clustering.labels_
        if is_plot:
            clusters_values = np.sort(np.unique(labels))
            n_clusters = len(clusters_values)
            print(f'no clusters: {n_clusters}')
            fig, axes = plt.subplots(1, 2, figsize=(30, 8), gridspec_kw={'width_ratios': (1, 3)})
            axes[0].scatter(x=X_normalized[:, 0], y=X_normalized[:, 1], c=labels,
                            cmap=plt.cm.get_cmap('jet', n_clusters))
            for i in clusters_values:
                xc, yc = X_normalized[labels == i, :].mean(axis=0)
                axes[0].text(xc, yc, str(i), fontsize=18, color='w',
                             path_effects=[pe.withStroke(linewidth=4, foreground="k")])
            utils.plot_dendrogram(clustering, truncate_mode="level", p=5, ax=axes[1])

        return X_normalized, labels

    def plot_temporal(self):
        fig, ax = plt.subplots(figsize=(30, 5), dpi=130)
        _, labels = self.cluster_2d()
        colors = utils.color_list(len(np.unique(labels)))
        ev_timestamps, ev_durations, ev_labels = self.mf.rd.reader.get_event_timestamps(block_index=0, seg_index=0,
                                                                                event_channel_index=1, t_start=None,
                                                                                t_stop=None)
        evt = (ev_timestamps - ev_timestamps[0]) / (1000 * 1000 * 3600)
        ax.set_xlim([evt[0], evt[-1]])
        ax.set_ylim([0, 2])
        ax.set_xlabel('Time [Hours]')
        for i in range(len(self.vfe.frames_ids) - 1):
            t1, t2 = evt[self.vfe.frames_ids[i]], evt[self.vfe.frames_ids[i + 1]]
            rect = patches.Rectangle((t1, 0), t2 - t1, 1, linewidth=1, edgecolor='none',
                                     facecolor=COLORS[labels[i]])
            ax.add_patch(rect)
