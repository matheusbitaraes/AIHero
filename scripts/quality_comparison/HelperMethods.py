import glob
import os
import pickle
from json import load as jload

import numpy as np
import pandas as pd
import scikit_posthocs as sph
import scipy as sp
from matplotlib import pyplot as plt, tri

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
        with open(f"{self.quality_config['WORK_DIR']}/config.json") as config_file:
            config = jload(config_file)
        ai_hero_service = AIHeroService(config)

        harmony_specs = build_request_body(request_body_bars=request_body_bars,
                                           random_evo_weights=random_evo_weights).melody_specs.harmony_specs
        # roda x vezes para GAN
        for i in range(self.quality_config['NUM_MELODIES_EACH']):
            gan_data = ai_hero_service.generate_GAN_compositions(harmony_specs, i)
            evo_data = ai_hero_service.generate_compositions(harmony_specs)
            gan_data.save_data(self.quality_config['GAN_DIRECTORY'], i)
            gan_data.export_as_midi(f"{self.quality_config['WORK_DIR']}/full_gan")
            evo_data.save_data(self.quality_config['EVO_DIRECTORY'], i)
            evo_data.export_as_midi(f"{self.quality_config['WORK_DIR']}/full_evo")

    def get_quality_measures_and_save(self):
        quality_metrics = {}
        gan_file = f'{self.quality_config["GAN_DIRECTORY"]}/gan'
        evo_file = f'{self.quality_config["EVO_DIRECTORY"]}/evo'
        test_file = f'{self.quality_config["TESTDATA_DIRECTORY"]}/test'
        datasetGAN = AIHeroData()
        datasetGAN.load_spr_from_checkpoint(self.quality_config["GAN_DIRECTORY"])
        datasetGAN.export_each_bar_as_midi(gan_file)
        datasetGAN.export_as_midi(f'{self.quality_config["WORK_DIR"]}/full_gan')
        datasetEVO = AIHeroData()
        datasetEVO.load_spr_from_checkpoint(self.quality_config["EVO_DIRECTORY"])
        datasetEVO.export_each_bar_as_midi(evo_file)
        datasetEVO.export_as_midi(f'{self.quality_config["WORK_DIR"]}/full_evo')

        self.training_data.ai_hero_data.export_each_bar_as_midi(test_file)

        # MGEval
        mgeval_model = MGEval()
        mgeval_gan = mgeval_model.eval(self.quality_config["GAN_DIRECTORY"], self.quality_config["TESTDATA_DIRECTORY"])
        mgeval_evo = mgeval_model.eval(self.quality_config["EVO_DIRECTORY"], self.quality_config["TESTDATA_DIRECTORY"])

        for metric in mgeval_gan.keys():
            quality_metrics['GAN_' + metric] = mgeval_gan[metric]
            quality_metrics['EVO_' + metric] = mgeval_evo[metric]

        # FID
        fid_model = FIDQualityModel()

        # get FID in comparison with training data
        print("calculating FID for dataset A")
        fid_array_GAN = fid_model.calculate_qualities(datasetGAN, self.training_data.ai_hero_data)
        quality_metrics["GAN_FID_array"] = fid_array_GAN

        print("calculating FID for dataset B")
        fid_array_EVO = fid_model.calculate_qualities(datasetEVO, self.training_data.ai_hero_data)
        quality_metrics["EVO_FID_array"] = fid_array_EVO

        # Save the dictionary to a file
        with open(f"{self.quality_config['WORK_DIR']}/{self.quality_config['QUALITY_DATA_NAME']}.pkl", 'wb') as f:
            pickle.dump(quality_metrics, f)
        return quality_metrics

    def generate_quality_plots(self):
        with open(f"{self.quality_config['WORK_DIR']}/{self.quality_config['QUALITY_DATA_NAME']}.pkl", 'rb') as f:
            quality_data = pickle.load(f)

        self._print_fid_stuff(quality_data)

        gan_kld = []
        evo_kld = []
        gan_oa = []
        evo_oa = []
        metrics = {
            'KLD': {
                'GAN': {

                },
                'EVO': {

                }
            },
            'OA': {
                'GAN': {

                },
                'EVO': {

                }
            }
        }
        for key in quality_data.keys():
            if isinstance(quality_data[key], dict):
                for inner_key in quality_data[key].keys():
                    # new_key = key.replace('GAN', '').replace('EVO', '')
                    # kkey = inner_key+new_key
                    # if kkey not in metrics.keys():
                    #     metrics[kkey] = {}
                    # metrics[kkey][key] = quality_data[key][inner_key]
                    if quality_data[key][inner_key] != 0:
                        if 'kld' in inner_key:
                            if 'GAN' in key:
                                gan_kld.append(quality_data[key][inner_key])
                                metrics['KLD']['GAN'][key.replace('GAN_', '').replace('EVO_', '')] = quality_data[key][
                                    inner_key]
                            if 'EVO' in key:
                                evo_kld.append(quality_data[key][inner_key])
                                metrics['KLD']['EVO'][key.replace('GAN_', '').replace('EVO_', '')] = quality_data[key][
                                    inner_key]
                        if 'oa' in inner_key:
                            if 'GAN' in key:
                                gan_oa.append(quality_data[key][inner_key])
                                metrics['OA']['GAN'][key.replace('GAN_', '').replace('EVO_', '')] = quality_data[key][
                                    inner_key]
                            if 'EVO' in key:
                                evo_oa.append(quality_data[key][inner_key])
                                metrics['OA']['EVO'][key.replace('GAN_', '').replace('EVO_', '')] = quality_data[key][
                                    inner_key]

        plt.figure(figsize=(20, 10))
        plt.title("KLDs")
        dic_gan = metrics['KLD']['GAN']
        dic_evo = metrics['KLD']['EVO']
        self._plot_chart(dic_gan, dic_evo, 'KLDs')

        plt.figure(figsize=(20, 10))
        plt.title("OAs")
        dic_gan = metrics['OA']['GAN']
        dic_evo = metrics['OA']['EVO']
        self._plot_chart(dic_gan, dic_evo, 'OAs')

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

    def _plot_chart(self, dic_gan, dic_evo, filename):
        proportions_gan = []
        proportions_evo = []
        labels = []
        for key in dic_gan.keys():
            # a, b = self._scale_values(dic_gan[key], dic_evo[key])
            proportions_gan.append(dic_gan[key])
            proportions_evo.append(dic_evo[key])
            labels.append(key)

        plt.bar(labels, proportions_gan, alpha=0.5, label='GAN')
        plt.bar(labels, proportions_evo, alpha=0.5, label='EVO')
        # plt.xticks(labels, labels, rotation=45)

        plt.legend()
        plt.savefig(f"{self.quality_config['WORK_DIR']}/plots/{filename}.png")

    def _print_fid_stuff(self, quality_data):

        FID_metrics = pd.DataFrame({
            'GAN_FID': quality_data['GAN_FID_array'],
            'EVO_FID': quality_data['EVO_FID_array']
        })

        # plots with original dataset
        fig, ax = plt.subplots()
        ax.set_title(f"FID boxplots ({len(FID_metrics['GAN_FID'])} sampels)")
        FID_metrics.boxplot(ax=ax)
        plt.savefig(f"{self.quality_config['WORK_DIR']}/plots/FID_boxpot.png")

        fida = FID_metrics['GAN_FID']
        fidb = FID_metrics['EVO_FID']
        print(f"GAN FID:{np.mean(fida)} +- {np.std(fida)}\n"
              f"EVO FID:{np.mean(fidb)} +- {np.std(fidb)}\n")

        # perform Nemenyi post-hoc test
        discr_data = np.array([fida, fidb])

        test = sph.posthoc_nemenyi_friedman(discr_data.T)
        print(f"p-values of posthoc Nemenyi Test (FID)")
        print(test)
