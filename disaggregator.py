import os, re
import numpy as np
import pandas as pd

from metrics import MetricsAccumulator

class Disaggregator():
    def __init__(self,
                 EVALUATION_DATA_PATH,
                 TARGET_APPLIANCE,
                 ON_POWER_THRESHOLD,
                 MAX_TARGET_POWER,
                 disagg_func,
                 disagg_kwargs,
                 remove_vampire_power=False,
                 pad_mains=True,
                 pad_appliance=False,
                 downsample_factor=1):
        self.EVALUATION_DATA_PATH = EVALUATION_DATA_PATH
        self.BUILDINGS = []
        self.TARGET_APPLIANCE = TARGET_APPLIANCE
        if TARGET_APPLIANCE == "dish washer":
            self.ON_POWER_THRESHOLD = 15
        else:
            self.ON_POWER_THRESHOLD = ON_POWER_THRESHOLD
        self.MAX_TARGET_POWER = MAX_TARGET_POWER
        self.PAD_WIDTH = 1536
        self.pad_mains = pad_mains
        self.pad_appliance = pad_appliance
        self.disagg_func = disagg_func
        self.disagg_kwargs = disagg_kwargs
        self.downsample_factor = downsample_factor
        self.metrics = MetricsAccumulator(self.ON_POWER_THRESHOLD, 4200)
        self._init_data(remove_vampire_power)

    def _init_data(self, remove_vampire_power):
        re_building_filename = re.compile("^((.*)_(.*))\\.pkl$")
        self.mains = {}
        self.appliance_y_true = {}
        for filename in os.listdir(self.EVALUATION_DATA_PATH):
            re_match = re_building_filename.match(filename)
            if re_match:
                building_i = re_match.group(1)
                mains, y_true = (
                    self._load_data(filename,
                        remove_vampire_power,
                        pad_mains=self.pad_mains,
                        pad_appliance=self.pad_appliance))
                if mains is None:
                    continue
                self.BUILDINGS.append(building_i)
                self.mains[building_i] = mains
                self.appliance_y_true[building_i] = y_true


    def _load_data(self, filename, remove_vampire_power, pad_mains=True, pad_appliance=False):
        # Load mains
        filename = os.path.join(self.EVALUATION_DATA_PATH, filename)
        df = pd.read_pickle(filename)

        if not self.TARGET_APPLIANCE in df:
            return None, None

        mains = df['mains'].values
        if remove_vampire_power:
            vampire_power = df['mains'].quantile(0.0002)
            mains = np.clip(mains-vampire_power, 0, None)
        if self.downsample_factor > 1:
            mains = self._resample_mains(mains)

        # Pad
        if pad_mains:
            mains = np.pad(
                mains, pad_width=(self.PAD_WIDTH, self.PAD_WIDTH), mode='constant')

        y_true = df[self.TARGET_APPLIANCE].values

        if pad_appliance:
            y_true = np.pad(
                y_true, pad_width=(self.PAD_WIDTH, self.PAD_WIDTH), mode='constant')

        return mains, y_true


    def _resample_mains(self, mains):
        mains_length_odd = len(mains)
        downsample_factor = self.downsample_factor
        n_samples_new = int(np.ceil(mains_length_odd/downsample_factor))
        mains_length_even = n_samples_new*downsample_factor
        mains_resampled = np.pad(mains, pad_width=(0, mains_length_even-mains_length_odd), mode='constant')
        mains_resampled = mains_resampled.reshape((n_samples_new, downsample_factor))
        mains_resampled[:, :] = mains_resampled.mean(axis=1)[:, np.newaxis]
        return mains_resampled.reshape((-1))[:mains_length_odd]

    def get_mains(self, building_i):
        if pad_mains:
            return self.mains[building_i][self.PAD_WIDTH:-self.PAD_WIDTH]
        else:
            return self.mains[building_i]


    def disaggregate(self, building_i, return_sequences=False):
        kwargs = dict(
            mains=self.mains[building_i],
            target=self.appliance_y_true[building_i],
            max_target_power=self.MAX_TARGET_POWER,
            building_i=building_i,
            return_sequences=return_sequences
        )
        kwargs.update(self.disagg_kwargs)

        if return_sequences:
            estimates, sequences = self.disagg_func(**kwargs)
        else:
            estimates = self.disagg_func(**kwargs)

        if self.pad_mains and not self.pad_appliance:
            estimates = estimates[self.PAD_WIDTH:-self.PAD_WIDTH]  # remove padding
        estimates = np.round(estimates).astype(int)

        if return_sequences:
            return estimates, sequences
        else:
            return estimates


    def calculate_metrics(self, return_sequences=False):
        scores = {}
        estimates = {}
        sequences = {}
        for building_i in self.BUILDINGS:
            mains = self.mains[building_i]
            y_true = self.appliance_y_true[building_i]
            if return_sequences:
                y_pred, y_sequences = self.disaggregate(building_i, return_sequences=True)
            else:
                y_pred = self.disaggregate(building_i)

            if self.pad_appliance:
                y_true = y_true[self.PAD_WIDTH:-self.PAD_WIDTH]  # remove padding
                y_pred = y_pred[self.PAD_WIDTH:-self.PAD_WIDTH]

            # Truncate
            n = min(len(y_true), len(y_pred))
            y_true = y_true[:n]
            y_pred = y_pred[:n]

            #np.savez("building_{}".format(building_i), y_true=y_true, y_pred=y_pred, mains=mains)
            scores[building_i] = self.metrics.run_metrics(y_true, y_pred, mains)

            if return_sequences:
                sequences[building_i] = y_sequences
                estimates[building_i] = y_pred

        if return_sequences:
            return scores, estimates, sequences
        else:
            return scores


    def save_disaggregation_data(self,
                                 model_name,
                                 estimates,
                                 sequences,
                                 SAVETO_DIR='./disaggregation_data/'):

        model_name = model_name.replace(" ", "_")

        SAVETO_DIR = os.path.join(SAVETO_DIR, model_name)

        for building_i in sequences:
            SAVETO_PartialPATH = os.path.join(SAVETO_DIR, str(building_i))
            os.makedirs(SAVETO_PartialPATH, exist_ok=True)
            
            SAVETO_PATHs = [os.path.join(SAVETO_PartialPATH, "estimate_windows.npz"),
                            os.path.join(SAVETO_PartialPATH, "estimate_average.npy"),
                            os.path.join(SAVETO_PartialPATH, "mains.npy"),
                            os.path.join(SAVETO_PartialPATH, "appliance.npy")]

            for SAVETO_PATH in SAVETO_PATHs:
                assert not os.path.exists(SAVETO_PATH), "ERROR: File {} already exists !!!!!!".format(SAVETO_PATH)

            np.savez(SAVETO_PATHs[0], **sequences[building_i])
            np.save(SAVETO_PATHs[1], estimates[building_i])
            np.save(SAVETO_PATHs[2], self.mains[building_i])
            np.save(SAVETO_PATHs[3], self.appliance_y_true[building_i])

            print("INFO: saved estimates windows to {}".format(SAVETO_PATHs[0]))
            print("INFO: saved estimates averages to {}".format(SAVETO_PATHs[1]))
            print("INFO: saved mains to {}".format(SAVETO_PATHs[2]))
            print("INFO: saved appliance_y_true to {}".format(SAVETO_PATHs[3]))


    def load_disaggregation_data(PATH):
        return np.load(PATH)



if __name__ == "__main__":
    import os, sys
    import jsonpickle
    import importlib
    import argparse

    DEFAULT_DATA_DIR = './input/evaluation_data'

    parser = argparse.ArgumentParser(description='Disaggregate experiment')
    parser.add_argument('experiment_name', help='experiment script name')
    parser.add_argument('appliance', metavar='appliance', help='target appliance')
    parser.add_argument('--nosequences', help='do not save all disaggregation windows', action='store_true')
    parser.add_argument('--id', metavar='id', type=int, help='experiment id', nargs='?')
    parser.add_argument('--modeldir', metavar='modeldir', help='model base folder', nargs='?')
    parser.add_argument('--modelfn', metavar='modelfn', help='model file name', nargs='?')
    parser.add_argument('--resultdir', metavar='resultdir', help='result folder', nargs='?')
    parser.add_argument('--data', metavar='data', help='data folder', nargs='?')
    #parser.add_argument('--downsample', metavar='factor', type=int, help='downsample mains by factor', default=1)

    args = parser.parse_args()

    experiment_name = args.experiment_name
    target_device = args.appliance
    experiment_id = args.id
    model_basedir = args.modeldir
    if model_basedir is None:
        model_basedir = experiment_name
    result_dir = args.resultdir
    if result_dir is None:
        result_dir = os.path.join("results", experiment_name)
    data_dir = args.data
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    model_name = args.modelfn
    if model_name is None:
        model_name = ""
    #downsample_factor = args.downsample

    def find_newest_id(path):
        def cast_to_int(s):
            try:
                return int(s)
            except ValueError:
                return 0
        ids = list(map(cast_to_int, os.listdir(path)))
        ids.sort()
        return ids[-1]

    if experiment_id is None:
        experiment_id = find_newest_id(os.path.join(model_basedir,target_device))

    experiment = importlib.import_module(experiment_name)

    disagg, _ = experiment.load_disaggregator(data_dir, '{}/{}/{}/{}'.format(model_basedir,target_device,experiment_id,model_name))

    result_dir = os.path.join(result_dir, str(experiment_id))

    os.makedirs(result_dir, exist_ok=True)

    if args.nosequences:
        results = disagg.calculate_metrics(return_sequences=False)
    else:
        results, estimates, sequences = disagg.calculate_metrics(return_sequences=True)
        disagg.save_disaggregation_data(target_device, estimates, sequences, SAVETO_DIR=result_dir)

    with open(os.path.join(result_dir, "{}.json".format(target_device)), "w") as f:
        f.write(jsonpickle.dumps(results))
