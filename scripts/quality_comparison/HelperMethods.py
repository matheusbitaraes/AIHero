import glob
import os
import pickle
from json import load as jload

import numpy as np
import pandas as pd
import scikit_posthocs as sph
from matplotlib import pyplot as plt

from src.GEN.data.GANTrainingData import GANTrainingData
from src.data.AIHeroData import AIHeroData
from src.quality.FID.FIDQualityModel import FIDQualityModel
from src.quality.mgeval.MGEval import MGEval
from src.service.AIHeroService import AIHeroService
from src.test.utils.test_utils import build_request_body


class HelperMethods:
    def __init__(self, config):
        self.quality_config = config
        with open(f"{self.quality_config['WORK_DIR']}/config.json") as config_file:
            self.module_config = jload(config_file)
        self.training_data = GANTrainingData(self.module_config)

    def generate_and_save_data(self, request_body_bars=12, random_evo_weights=False):

        harmony_specs = build_request_body(request_body_bars=request_body_bars,
                                           random_evo_weights=random_evo_weights).melody_specs.harmony_specs

        for model in self.quality_config['MODELS']:
            with open(f"{self.quality_config['WORK_DIR']}/{model['CONFIG_FILE_NAME']}") as config_file:
                config = jload(config_file)
            service = AIHeroService(config)
            print(f"Generating {model['NAME']} data...")
            if 'EVO' in model['NAME']:
                data = service.generate_compositions(harmony_specs)
            else:
                data = service.generate_GEN_compositions(harmony_specs)
            data.save_data(model['DIRECTORY'])
            data.export_as_midi(f"{self.quality_config['WORK_DIR']}/{model['NAME']}_full")
            service.clear()

    def get_quality_measures_and_save(self):
        quality_metrics = {}
        fid_model = FIDQualityModel()
        mgeval = MGEval()

        generated_models = [model for model in self.quality_config['MODELS'] if model['TYPE'] == 'GENERATED']
        test_model = [model for model in self.quality_config['MODELS'] if model['TYPE'] == 'TEST'][0]
        self.training_data.ai_hero_data.export_each_bar_as_midi(f"{test_model['DIRECTORY']}/test_data")

        # load generated bars and export each bar individually as midi
        for model in generated_models:
            file = f"{model['DIRECTORY']}/generated_data"
            dataset = AIHeroData()
            dataset.load_spr_from_checkpoint(model['DIRECTORY'])
            dataset.export_each_bar_as_midi(file)

            result = mgeval.eval(model['DIRECTORY'], test_model['DIRECTORY'])

            # iterate over metrics and store in quality_metrics dict
            for metric in result.keys():
                quality_metrics[model['NAME'] + '_' + metric] = result[metric]

            print(f"calculating FID for dataset {model['NAME']}")
            fid_array = fid_model.calculate_qualities(dataset, self.training_data.ai_hero_data)
            quality_metrics[model["NAME"] + "_FID_array"] = fid_array

        # Save the dictionary to a file
        with open(f"{self.quality_config['WORK_DIR']}/{self.quality_config['QUALITY_DATA_NAME']}.pkl", 'wb') as f:
            pickle.dump(quality_metrics, f)
        return quality_metrics

    def _add_metric(self, quality_data, data, key, inner_key, metrics, model_names):
        for metric in metrics:
            if metric in inner_key:
                for name in model_names:
                    if name + '_' in key:

                        # gambiarra
                        if not (name == 'EVO' and 'ALT' in key):
                            data[metric][name][key.replace(name + '_', '')] = quality_data[key][inner_key]

    def _get_distance_metrics(self, quality_data):
        metrics_list = ['kld_intra1_inter',
                        'oa_intra1_inter',
                        'kld_intra2_inter',
                        'oa_intra2_inter',
                        'kld_intra1_intra2',
                        'oa_intra1_intra2']
        models_list = ['GAN', 'EVO', 'EVO_ALT', 'LSTM']
        metrics_data = {}
        for metric in metrics_list:
            metrics_data[metric] = {}
            for model in models_list:
                metrics_data[metric][model] = {}
        for key in quality_data.keys():
            if isinstance(quality_data[key], dict):
                for inner_key in quality_data[key].keys():
                    if quality_data[key][inner_key] != 0:
                        self._add_metric(quality_data, metrics_data, key, inner_key, metrics_list, models_list)
        return metrics_data

    def generate_quality_plots(self):
        with open(f"{self.quality_config['WORK_DIR']}/{self.quality_config['QUALITY_DATA_NAME']}.pkl", 'rb') as f:
            quality_data = pickle.load(f)

        models_comparison_a = ['GAN', 'EVO', 'LSTM']
        models_comparison_b = ['EVO', 'EVO_ALT']
        self._print_mgeval_stuff(quality_data, models_comparison_a, title='exp 2')
        self._print_fid_stuff(quality_data, models_comparison_a, title='exp 2')
        self._print_mgeval_stuff(quality_data, models_comparison_b, title='exp 3')
        self._print_fid_stuff(quality_data, models_comparison_b, title='exp 3')

    def delete_files_with_pattern(self, dir, pattern):
        files = glob.glob(os.path.join(dir, pattern))

        # Iterate over the list of files
        for file in files:
            # Delete the file
            os.remove(file)

    def _scale_values(self, x, y):
        # Find the minimum and maximum values
        max_value = max(x, y)

        # Scale the values
        a = x / max_value
        b = y / max_value

        return a, b

    def _plot_chart(self, dic_gan, dic_evo, dic_evo_alt, dic_lstm, filename):
        proportions_gan = []
        proportions_evo = []
        proportions_evoalt = []
        proportions_lstm = []
        labels = []
        for key in dic_gan.keys():
            # a, b = self._scale_values(dic_gan[key], dic_evo[key])
            proportions_gan.append(dic_gan[key])
            proportions_evo.append(dic_evo[key])
            proportions_evoalt.append(dic_evo_alt[key])
            proportions_lstm.append(dic_lstm[key])
            labels.append(key)

        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        ax = plt.subplot()
        ax.bar(x, proportions_gan, width, alpha=0.5, label='GAN')
        ax.bar(x + width, proportions_evo, width, alpha=0.5, label='EVO')
        ax.bar(x + width * 2, proportions_evoalt, width, alpha=0.5, label='EVO_ALT')
        ax.bar(x + width * 3, proportions_lstm, width, alpha=0.5, label='LSTM')
        ax.set_xticks(x, labels)

        plt.legend()
        plt.savefig(f"{self.quality_config['WORK_DIR']}/plots/{filename}.png")

    def _print_fid_stuff(self, quality_data, models_list, title):
        fid_data = {
            'GAN': quality_data['GAN_FID_array'],
            'EVO': quality_data['EVO_FID_array'],
            'EVO_ALT': quality_data['EVO_ALT_FID_array'],
            'LSTM': quality_data['LSTM_FID_array']
        }

        # filter fid_data according to models_list:
        fid_data = {key: fid_data[key] for key in models_list}

        FID_metrics = pd.DataFrame(fid_data)

        fids = []
        text = 'Média e desvio padrão:\n'
        for model_name in models_list:
            fids.append(FID_metrics[model_name])
            text += f"{model_name}: {np.mean(FID_metrics[model_name]):.3f}+-{np.std(FID_metrics[model_name]):.3f}\n"
            print(
                f"{title}\n \\textit{{{model_name}}} & {np.mean(FID_metrics[model_name]):.3f}\pm{np.std(FID_metrics[model_name]):.3f}")

        # perform Nemenyi post-hoc test
        discr_data = np.array(fids)

        test = sph.posthoc_nemenyi_friedman(discr_data.T)
        print(f"p-values of posthoc Nemenyi Test (FID)")
        test.columns = models_list
        test.index = models_list
        print(test.round(4))

        # plots with original dataset
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.set_title(f"{title} - Boxplots de FIDs ({len(FID_metrics['EVO'])} amostras)")
        ax.text(1.05, 0.5, text + '\n\n' + 'p-valores do teste de Nemenyi:\n' + test.round(4).to_string(), #.to_latex(),
                transform=ax.transAxes,
                fontsize=8)
        plt.subplots_adjust(right=.7)   
        FID_metrics.boxplot(ax=ax)
        plt.savefig(f"{self.quality_config['WORK_DIR']}/plots/{title}_FID_boxpot.png")

    def _print_mgeval_stuff(self, quality_data, models_list, title):
        distance_metrics = self._get_distance_metrics(quality_data)
        selected_metrics = ['total_used_pitch', 'pitch_range', 'avg_IOI', 'total_used_note',
                            'total_pitch_class_histogram', 'note_length_hist', 'note_length_transition_matrix',
                            'pitch_class_transition_matrix']
        # self._metric_comparison_plots(distance_metrics['kld_intra1_inter'], title=f'{title}',
        #                               filename=f'{title}_KLD_1',
        #                               selected_metrics=selected_metrics, models=models_list)
        # self._metric_comparison_plots(distance_metrics['oa_intra1_inter'], title=f'{title}',
        #                               filename=f'{title}_OA_1',
        #                               selected_metrics=selected_metrics,
        #                               models=models_list)
        # self._metric_comparison_plots(distance_metrics['kld_intra2_inter'], title=f'{title}',
        #                               filename=f'{title}_KLD_2',
        #                               selected_metrics=selected_metrics, models=models_list)
        # self._metric_comparison_plots(distance_metrics['oa_intra2_inter'], title=f'{title}',
        #                               filename=f'{title}_OA_2',
        #                               selected_metrics=selected_metrics,
        #                               models=models_list)
        self._metric_comparison_plots(distance_metrics['kld_intra1_intra2'], title=f'{title}',
                                      filename=f'{title}_KLD_3',
                                      selected_metrics=selected_metrics, models=models_list)
        self._metric_comparison_plots(distance_metrics['oa_intra1_intra2'], title=f'{title}',
                                      filename=f'{title}_OA_3',
                                      selected_metrics=selected_metrics,
                                      models=models_list)

    def _metric_comparison_plots(self, metrics, title, filename, selected_metrics,
                                 models=['GAN', 'EVO', 'EVO_ALT', 'LSTM']):
        for metric in selected_metrics:
            data = pd.DataFrame()
            for model in models:
                data[model] = metrics[model][metric]
            self._make_boxplot_and_save(data, title=f"{title} - {metric} ({data.shape[0]} amostras)",
                                        filename=f"{filename}_{metric}")

    def _make_boxplot_and_save(self, dataset, title, filename):
        data = []
        text = 'Média e desvio padrão:\n'
        for model_name in dataset.keys():
            data.append(dataset[model_name])
            text += f"{model_name}: {np.mean(dataset[model_name]):.3f}+-{np.std(dataset[model_name]):.3f}\n"
            print(
                f"{title}\n \\textit{{{model_name}}} & {np.mean(dataset[model_name]):.3f}\pm{np.std(dataset[model_name]):.3f}")

        # perform Nemenyi post-hoc test
        discr_data = np.array(data)

        test = sph.posthoc_nemenyi_friedman(discr_data.T)
        print(f"p-values of posthoc Nemenyi Test ({filename})")
        test.columns = dataset.keys()
        test.index = dataset.keys()
        print('p-valores do teste de Nemenyi')
        print(test.round(4))

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.set_title(title)
        ax.text(1.05, 0.5, text + '\n\n' + 'p-valores do teste de Nemenyi:\n' + test.round(4).to_string(), #.to_latex(),
                transform=ax.transAxes,
                # transform=plt.gcf().transFigure,
                fontsize=8)
        plt.subplots_adjust(right=.7)
        dataset.plot.box(ax=ax, widths=0.3)
        plt.savefig(f"{self.quality_config['WORK_DIR']}/plots/{filename}.png")

        # print( f'\\begin{{figure}}[htb]\n'
        #        f'\t\centering\n'
        #        f'\t\includegraphics[width=\\textwidth]{{figures/{filename}.png}}\n'
        #        f'\t\caption{{{title.replace("_", " ")}}}\n'
        #        f'\t\label{{{filename}}}\n'
        #        f'\end{{figure}}\n'
        # )
