import copy
import glob
import os

import numpy as np
from sklearn.model_selection import LeaveOneOut
from random import sample

from src.quality.mgeval import utils, core


class MGEval:
    def __init__(self):
        self.num_bar = 1
        self.enable_bar_metrics = False
        self.calculate_intraset_metrics = False
        self._num_evaluation_samples = 100
        self._batch_percentage = 0.1

    def _initialize(self, num_samples):
        evalset = {
            'total_used_pitch': np.zeros((num_samples, 1))
            , 'pitch_range': np.zeros((num_samples, 1))
            , 'avg_pitch_shift': np.zeros((num_samples, 1))
            , 'avg_IOI': np.zeros((num_samples, 1))
            , 'total_used_note': np.zeros((num_samples, 1))
            , 'bar_used_pitch': np.zeros((num_samples, self.num_bar, 1))
            , 'bar_used_note': np.zeros((num_samples, self.num_bar, 1))
            , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
            , 'bar_pitch_class_histogram': np.zeros((num_samples, self.num_bar, 12))
            , 'note_length_hist': np.zeros((num_samples, 12))
            , 'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12))
            , 'note_length_transition_matrix': np.zeros((num_samples, 12, 12))
        }

        self.bar_metrics = ['bar_used_pitch', 'bar_used_note', 'bar_pitch_class_histogram']
        self.single_arg_metrics = (
            ['total_used_pitch'
                , 'avg_IOI'
                , 'total_pitch_class_histogram'
                , 'pitch_range'
             ])

        if not self.enable_bar_metrics:
            for metric in self.bar_metrics:
                evalset.pop(metric)

        self.set1_eval = copy.deepcopy(evalset)
        self.set2_eval = copy.deepcopy(evalset)

        self.metrics_list = evalset.keys()

    def _expand_set(self, _set, num_samples):
        while len(_set) < num_samples:
            _set.extend(_set[:-1])

        return _set[:num_samples]

    def _calculate_metrics(self, set1, set2):
        num_samples = len(set1)
        self._initialize(num_samples)

        sets = [(set1, self.set1_eval), (set2, self.set2_eval)]

        # Extract Fetures
        for _set, _set_eval in sets:
            for i in range(0, num_samples):
                feature = core.extract_feature(_set[i])
                for metric in self.metrics_list:
                    evaluator = getattr(core.metrics(), metric)
                    if metric in self.single_arg_metrics:
                        tmp = evaluator(feature)
                    elif metric in self.bar_metrics:
                        tmp = evaluator(feature, 1, self.num_bar)
                    else:
                        tmp = evaluator(feature, 1)
                    _set_eval[metric][i] = tmp

        set1_intra, set2_intra = self._calculate_intra_set_metrics(num_samples)
        sets_inter = self._calculate_inter_set_metrics(num_samples)

        plot_set1_intra = np.transpose(
            set1_intra, (1, 0, 2)).reshape(len(self.metrics_list), -1)
        plot_set2_intra = np.transpose(
            set2_intra, (1, 0, 2)).reshape(len(self.metrics_list), -1)
        plot_sets_inter = np.transpose(
            sets_inter, (1, 0, 2)).reshape(len(self.metrics_list), -1)

        distances = {}
        for i, metric in enumerate(self.metrics_list):
            kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
            ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
            kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
            ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])
            kl3 = utils.kl_dist(plot_set1_intra[i], plot_set2_intra[i])
            ol3 = utils.overlap_area(plot_set1_intra[i], plot_set2_intra[i])

            distances[metric] = {
                'kld_intra1_inter': kl1,
                'oa_intra1_inter': ol1,
                'kld_intra2_inter': kl2,
                'oa_intra2_inter': ol2,
                'kld_intra1_intra2': kl3,
                'oa_intra1_intra2': ol3
            }
        return distances

    def _calculate_metrics_in_batches(self, set1, set2):
        sample_size = max(3, int(self._batch_percentage * len(set1)))
        batch_distances = []
        for _ in range(self._num_evaluation_samples):
            total_percent = (_ + 1) / self._num_evaluation_samples * 100
            set1_sample = sample(set1, sample_size)
            set2_sample = sample(set2, sample_size)
            batch_distances.append(self._calculate_metrics(set1_sample, set2_sample))
            print(f'\r MGEVAL CALCULATION PROGRESS: [{"#" * int(total_percent):<100}] {total_percent:.1f}%', end='')
        print('')

        output = self._transform_batch_distances(batch_distances)

        return output

    def eval(self, dataset1dir, dataset2dir):
        set1 = glob.glob(os.path.join(dataset1dir, '*mid'))
        set2 = glob.glob(os.path.join(dataset2dir, '*mid'))

        # fazer em batches, sampleando aleatoriamente as duas amostras e realizando os caluclos
        output = self._calculate_metrics_in_batches(set1, set2)

        return output

    def _calculate_intra_set_metrics(self, num_samples):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(num_samples))
        set1_intra = np.zeros((num_samples, len(self.metrics_list), num_samples - 1))
        set2_intra = np.zeros((num_samples, len(self.metrics_list), num_samples - 1))

        if num_samples <= 1:
            return set1_intra, set2_intra

        # Calculate Intra-set Metrics
        for i, metric in enumerate(self.metrics_list):
            for train_index, test_index in loo.split(np.arange(num_samples)):
                set1_intra[test_index[0]][i] = utils.c_dist(
                    self.set1_eval[metric][test_index], self.set1_eval[metric][train_index])
                set2_intra[test_index[0]][i] = utils.c_dist(
                    self.set2_eval[metric][test_index], self.set2_eval[metric][train_index])

        return set1_intra, set2_intra

    def _calculate_inter_set_metrics(self, num_samples):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(num_samples))
        sets_inter = np.zeros((num_samples, len(self.metrics_list), num_samples))

        # Calculate Inter-set Metrics
        for i, metric in enumerate(self.metrics_list):
            for train_index, test_index in loo.split(np.arange(num_samples)):
                sets_inter[test_index[0]][i] = utils.c_dist(self.set1_eval[metric][test_index], self.set2_eval[metric])

        return sets_inter

    def _transform_batch_distances(self, batch_distances):
        # transform the list of dictionaries into a dictionary of lists
        output = {k: [dic[k] for dic in batch_distances] for k in batch_distances[0]}

        for metric in output.keys():
            x = output[metric]
            transformed_metric = {k: [dic[k] for dic in x] for k in x[0]}
            output[metric] = transformed_metric

        return output
