import somoclu
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skew, entropy, zscore, mode, mannwhitneyu, ttest_ind
from scipy.spatial import distance
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans


class SOMClusterer:
    def __init__(self, V, start_indices, t, is_zscore=False, is_debug=True):
        self.V = V
        self.start_indices = start_indices
        self.som = None
        self.is_zscore = is_zscore
        self.is_debug = is_debug
        self.sig_df = self.create_sig_df(t, start_indices)

    def train(self, t=None, n_clusters=(8,)):
        X_pca = self.run_pca()
        self.train_som(X_pca)
        self.save_analysis_cache_data()
        self.save_clustering_results(n_clusters, X_pca, v, t)

    def save_clustering_results(self, n_clusters, X_pca, v, t):
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]
        for n in n_clusters:
            cluster_algo = self.cluster_som(n)
            self.update_clusters(t)
            self.find_distances_to_cluster_centers(X_pca, cluster_algo)
            self.plot_clusters(v, is_save=True, only_relevant=False)
            self.plot_clusters_histogram(is_save=True)
            self.plot_clusters_across_night(t, is_save=True)
            self.plot_label_summary(t, is_save=True)
        self.save_analysis_cache_data()

    def update_clusters(self, t):
        labels = self.som.clusters[self.som.bmus[:, 1], self.som.bmus[:, 0]]
        self.print(f'Num labels: {len(np.unique(labels))}')
        self.sig_df['label'] = labels
        self.sig_df['calc_end'] = self.sig_df.start + (self.w - self.noverlap)
        self.sig_df['calc_label'] = self.get_label_by_majority_vote(self.sig_df.label.to_numpy())
        self.relevant_labels_scores = self.find_relevant_labels(t)

    def find_distances_to_cluster_centers(self, X_pca, cluster_algo):
        for i in tqdm(range(len(self.sig_df))):
            label_id = int(self.sig_df.iloc[i].label)
            self.sig_df.loc[i, 'distance'] = distance.euclidean(X_pca[i, :], cluster_algo.cluster_centers_[label_id, :])

    def get_label_by_majority_vote(self, labels):
        x = []
        idx_back = int(self.w / (self.w - self.noverlap) - 1)
        for i in tqdm(range(len(labels))):
            md = pd.Series(labels[i - idx_back:i]).mode().values
            x.append(md[0] if len(md) == 1 else np.nan)
        return x

    def get_sleep_cycle(self, cycle_id, t, v, is_plot=False) -> (np.ndarray, np.ndarray, int):
        assert cycle_id < len(self.parser.sc), f'Cycle {cycle_id} is out of sleep cycles range; ' \
                                                f'Number of sleep cycles: {len(self.parser.sc)}'
        cycle_times = self.parser.sc.iloc[cycle_id]
        start_id = np.argmin(np.abs(t - cycle_times.on))
        end_id = np.argmin(np.abs(t - cycle_times.off))
        v = v.flatten()[start_id:end_id]
        t = t[start_id:end_id]
        if is_plot:
            plt.figure(figsize=(25,5))
            plt.plot(t, v)
            plt.title(f'Cycle {cycle_id}')
        return v, t, start_id

    def predict_signal(self, S, start_indices, is_plot=False, t=None):
        assert not is_plot or is_plot and t is not None, 'must pass time vector for label plotting'
        S = self.pca.transform(S)[:, :self.n_pca]
        if self.is_zscore:
            S = zscore(S)
        map_idx = self.som.get_bmus(self.som.get_surface_state(S))
        labels = self.som.clusters[map_idx[:, 1], map_idx[:, 0]]
        labels = self.get_label_by_majority_vote(labels)

        if is_plot:
            self.plot_signal_predictions(sig, labels, start_indices, t)

        return labels

    def run_pca(self):
        """PCA for feature reduction. Take only the PCs that sum to 95% explained variance"""
        self.print(f'runing PCA over {self.V.shape[1]} features...')
        self.pca = PCA(n_components=30)
        data = self.pca.fit_transform(self.V)
        c = 0
        i = -1
        for i, pc in enumerate(self.pca.explained_variance_ratio_):
            c += pc
            if c >= 0.97:
                break

        self.n_pca = i + 1
        self.print(f'Number of features after PCA: {self.n_pca}')
        data = data[:, :self.n_pca]
        self.print(f'New data shape: {data.shape}')
        return data

    def train_som(self, X, n_rows=100, n_columns=160):
        self.print('Start SOM train...')
        if self.is_zscore:
            X = zscore(X)
        self.som = somoclu.Somoclu(n_columns, n_rows, data=X)
        self.som.train()

    def cluster_som(self, n_clusters=None, is_save=True):
        self.print('Start SOM clustering...')
        # algorithm = DBSCAN(eps=0.2, min_samples=10)
        algorithm = KMeans(n_clusters=n_clusters, random_state=0)
        self.som.cluster(algorithm=algorithm)
        self.current_n_clusters = n_clusters
        self.cluster_plots_dir.mkdir(exist_ok=True)
        filename = f'{self.cluster_plots_dir}/view_umatrix.png' if is_save else None
        self.som.view_umatrix(bestmatches=True, filename=filename)
        return algorithm

    def print(self, s):
        if self.is_debug:
            print(s)


