#!/bin/env python3

from __future__ import print_function, division, absolute_import

import os, pickle
from collections import OrderedDict
import gc
import ipdb

import pandas as pd
import numpy as np

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("AGG")
import matplotlib.pyplot as plt

from neuralnilm.data.realaggregateactivitysources import RealAggregateActivityData, BalancedActivityRealAggregateSource, BalancedActivityAugmentedAggregateSource, RandomizedSequentialSource, BalancedBuildingRealAggregateSource
from neuralnilm.utils import check_windows

import nilmtk
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
from nilmtk.utils import timedelta64_to_secs

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

from sacred import Ingredient, Experiment
dataset = Ingredient('dataset')
standalone = Experiment(ingredients=[dataset])

def select_windows(TRAIN_BUILDINGS, TEST_BUILDINGS, WINDOWS):
    windows = {dataset: {} for dataset in list(WINDOWS.keys())}

    def copy_window(dataset, fold, i):
        if fold in WINDOWS[dataset] and i in WINDOWS[dataset][fold]:
            windows[dataset][fold][str(i)] = WINDOWS[dataset][fold][i]

    for dataset in WINDOWS.keys():
        windows[dataset] = {fold: {} for fold in list(WINDOWS[dataset].keys())}
        for i in TRAIN_BUILDINGS[dataset]:
            copy_window(dataset, 'train', i)
            copy_window(dataset, 'unseen_activations_of_seen_appliances', i)
        for i in TEST_BUILDINGS[dataset]:
            copy_window(dataset, 'unseen_appliances', i)

    return windows


@dataset.config
def dataset_config():
    NILMTK_EXPORT_PATH = "input/nilmtk_export_URE__"
    ACTIVITY_DATA_PATH = "input/real_aggregate_activity_data_URE__"
    EVALUATION_DATA_PATH = "input/evaluation_data_UR__K"
    TARGET_APPLIANCE = 'washing machine'
    APPLIANCE_ALIASES = {
        'UKDALE': {
            "fridge" :          ["fridge", "freezer", "fridge freezer"],
            "kettle" :          ["kettle"],
            "microwave" :       ["microwave"],
            "dish washer" :     ["dish washer"],
            "washing machine" : ["washing machine", "washer dryer"],
            "tumble dryer" :    ["tumble dryer"],
            "television" :      ["television"]
        },
        'REFIT': {
            "fridge" :          ["fridge", "freezer", "fridge freezer"],
            "kettle" :          ["kettle"],
            "microwave" :       ["microwave"],
            "dish washer" :     ["dish washer"],
            "washing machine" : ["washing machine", "washer dryer"],
            "tumble dryer" :    ["tumble dryer"],
            "television" :      ["television"]
        },
        'ECO': {
            "fridge" :          ["fridge", "freezer", "fridge freezer"],
            "kettle" :          ["kettle"],
            "microwave" :       ["microwave"],
            "dish washer" :     ["dish washer"],
            "washing machine" : ["washing machine"],
            "tumble dryer" :    [],
            "television" :      ["television"]
        },
        'DRED': {
            "fridge" :          ["fridge"],
            "kettle" :          [],
            "microwave" :       ["microwave"],
            "dish washer" :     [],
            "washing machine" : ["washing machine"],
            "tumble dryer" :    [],
            "television" :      ["television"]
        },
        'ESHL': {
            "fridge" :          ["fridge", "freezer"],
            "kettle" :          ["kettle"],
            "microwave" :       ["microwave"],
            "dish washer" :     ["dish washer"],
            "washing machine" : ["washing machine"],
            "tumble dryer" :    ["tumble dryer"],
            "television" :      ["television"]
        },
    }
    ON_POWER_THRESHOLD = {
        "fridge" :          30,
        "kettle" :         400,
        "microwave" :      200,
        "dish washer" :     15,
        "washing machine" : 20,
        "tumble dryer" :    20,
        "television" :      15
    }[TARGET_APPLIANCE]
    MIN_ON_DURATION = {
        "fridge" :            60,
        "kettle" :            12,
        "microwave" :         12,
        "dish washer" :     1800,
        "washing machine" : 1800,
        "tumble dryer" :    1800,
        "television" :       300
    }[TARGET_APPLIANCE]
    MIN_OFF_DURATION = {
        "fridge" :            12,
        "kettle" :             0,
        "microwave" :         72,
        "dish washer" :     1800,
        "washing machine" :  900,
        "tumble dryer" :     360,
        "television" :       300
    }[TARGET_APPLIANCE]
    SAMPLE_PERIOD = 6
    DATASET_PATHS = {
        'UKDALE': os.path.join('/home/aifb', 'nilmdata', 'ukdale2017.h5'),
        'REFIT': os.path.join('/home/aifb', 'nilmdata', 'refit.h5'),
        'ECO': os.path.join('/home/aifb', 'nilmdata', 'eco.h5'),
        'DRED': os.path.join('/home/aifb', 'nilmdata', 'dred.h5'),
        'ESHL': os.path.join('/home/aifb', 'nilmdata', 'eshldtestv6.h5'),
    }
    WINDOWS = {
        'UKDALE': {
            'train': {
                1: [(None, '2017-03-29'), ('2017-04-26', None)],
                2: [(None, '2013-09-12'), ('2013-10-10', None)],
                4: [(None, '2013-09-03')], # window after eval section does not contain any data
            },
            'unseen_activations_of_seen_appliances' : {
                1: [("2017-04-24 02:47:06", "2017-04-26 02:47:07")],
                2: [("2013-09-12 14:05:00", "2013-09-14 14:05:01")],
                4: [("2013-09-13 12:18:36", "2013-09-15 12:18:37")],
            },
            'unseen_appliances': { # not used: test data is exported in ExportTestData.ipynb
                5: [(None, None),]
            }
        },
        'REFIT': {
            'train': {
                 1: [(None, None)],
                 2: [(None, None)],
                 3: [(None, None)],
                 4: [(None, None)],
                 5: [(None, None)],
                 6: [(None, None)],
                 7: [(None, None)],
                 8: [(None, None)],
                10: [(None, None)],
                12: [(None, None)],
                13: [(None, None)],
                15: [(None, None)],
                16: [(None, None)],
                17: [(None, None)],
                18: [(None, None)],
            },
            'unseen_activations_of_seen_appliances' : {
                 1 : [("2015-07-05 02:10:12", "2015-07-07 02:10:13")],
                 2 : [("2015-05-02 03:10:30", "2015-05-04 03:10:31")],
                 3 : [("2015-05-29 12:46:54", "2015-05-31 12:46:55")],
                 4 : [("2015-05-25 04:42:18", "2015-05-27 04:42:19")],
                 5 : [("2015-01-23 14:47:48", "2015-01-25 14:47:49")],
                 6 : [("2015-05-01 11:46:18", "2015-05-03 10:16:43")],
                 7 : [("2015-05-19 16:19:42", "2015-05-21 16:19:43")],
                 8 : [("2015-05-04 11:00:06", "2015-05-06 11:00:07")],
                10 : [("2015-05-05 05:57:30", "2015-05-07 05:57:31")],
                12 : [("2015-03-07 10:03:18", "2015-03-09 10:03:19")],
                13 : [("2015-04-26 06:47:18", "2015-04-28 06:47:19")],
                15 : [("2015-03-27 13:41:30", "2015-03-29 13:41:31")],
                16 : [("2014-12-02 04:12:36", "2014-12-04 04:12:37")],
                17 : [("2014-12-13 11:36:30", "2014-12-15 11:36:31")],
                18 : [("2015-04-19 20:20:24", "2015-04-21 20:20:25")],
            },
            'unseen_appliances': { # not used: test data is exported in ExportTestData.ipynb
                9: [(None, None),],
                14: [(None, None),],
                19: [(None, None),],
            }
        },
        'ECO': {
            'train': {
                1: [('2012-07-01', None)],
                2: [(None, None)],
                3: [(None, None)],
                4: [(None, None)],
                5: [(None, None)],
                6: [(None, None)],
            },
            'unseen_activations_of_seen_appliances' : {
                1: [("2012-12-18 16:14:36", "2012-12-20 16:14:37")],
                2: [("2012-09-28 17:36:42", "2012-09-30 17:36:43")],
                3: [("2013-01-05 13:07:36", "2013-01-07 13:07:37")],
                4: [("2012-11-26 15:44:48", "2012-11-28 15:44:49")],
                5: [("2012-07-03 06:52:24", "2012-07-05 06:52:25")],
                6: [("2012-07-11 03:21:30", "2012-07-13 03:21:31")],
            },
            'unseen_appliances': { # not used: test data is exported in ExportTestData.ipynb
            }
        },
        'DRED': {
            'train': {
            },
            'unseen_activations_of_seen_appliances' : {
            },
            'unseen_appliances': { # not used: test data is exported in ExportTestData.ipynb
                #1: [(None, None)],
            }
        },
        'ESHL': {
            'train': {
            },
            'unseen_activations_of_seen_appliances' : {
                1: [("2015-11-21 07:29:18", "2015-11-23 07:29:19"),]
            },
            'unseen_appliances': { # not used: test data is exported in ExportTestData.ipynb
                1: [(None, None),],
                2: [(None, None),],
            }
        },
    }
    
    if TARGET_APPLIANCE == 'television':
        TRAIN_BUILDINGS = {
            'UKDALE': [1],
            'REFIT': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18],
            'ECO': [2],
            'DRED': [1],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'microwave':
        TRAIN_BUILDINGS = {
            'UKDALE': [1, 2],
            'REFIT': [2, 3, 4, 5, 6, 8, 10, 12, 13, 16, 17, 18],
            'ECO': [4, 5],
            'DRED': [1],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'tumble dryer':
        TRAIN_BUILDINGS = {
            'UKDALE': [1],
            'REFIT': [3, 5, 7, 8, 12, 13, 16],
            'ECO': [],
            'DRED': [],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'fridge':
        TRAIN_BUILDINGS = {
            'UKDALE': [1, 2, 4],
            'REFIT': [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 18],
            'ECO': [1, 2, 3, 4, 5, 6],
            'DRED': [1],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'kettle':
        TRAIN_BUILDINGS = {
            'UKDALE': [1, 2, 4],
            'REFIT': [2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 18],
            'ECO': [1, 2], # building 3 has too less data for kettle
            'DRED': [],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'washing machine':
        TRAIN_BUILDINGS = {
            'UKDALE': [1, 2],
            'REFIT': [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 18],
            'ECO': [1],
            'DRED': [1],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    elif TARGET_APPLIANCE == 'dish washer':
        TRAIN_BUILDINGS = {
            'UKDALE': [1, 2],
            'REFIT': [1, 2, 3, 5, 6, 7, 9, 10, 12, 15, 17],
            'ECO': [2],
            'DRED': [],
            'ESHL': []
            }
        TEST_BUILDINGS = {
            'UKDALE': [5],
            'REFIT': [9, 14, 19],
            'ECO': [],
            'DRED': [],
            'ESHL': [1, 2],
            }
    
    WINDOWS = select_windows(
                TRAIN_BUILDINGS,
                TEST_BUILDINGS,
                WINDOWS)
    
    VAMPIRE_POWER = {
        'UK-DALE_building_1': 73.7,
        'UK-DALE_building_2': 95.9,
        'UK-DALE_building_4': 131.0,
        'REFIT_building_1': 134.0,
        'REFIT_building_2': 54.0,
        'REFIT_building_3': 83.0,
        'REFIT_building_4': 47.0,
        'REFIT_building_5': 110.0,
        'REFIT_building_6': 174.0,
        'REFIT_building_7': 89.0,
        'REFIT_building_8': 163.0,
        'REFIT_building_10': 140.0,
        'REFIT_building_12': 117.0,
        'REFIT_building_13': 84.0,
        'REFIT_building_15': 123.0,
        'REFIT_building_16': 74.0,
        'REFIT_building_17': 63.0,
        'REFIT_building_18': 128.5,
        'ECO_building_1': 46.6,
        'ECO_building_2': 34.0,
        'ECO_building_3': 31.6,
        'ECO_building_4': 152.8,
        'ECO_building_5': 133.9,
        'ECO_building_6': 62.3
    }


def select_train_windows(WINDOWS):
    windows = {dataset: {} for dataset in list(WINDOWS.keys())}

    for dataset, folds_and_buildings in WINDOWS.items():
        for fold, buildings in folds_and_buildings.items():
            if fold in ['train', 'unseen_activations_of_seen_appliances']:
                windows[dataset][fold] = {}
                for i in buildings:
                    windows[dataset][fold][int(i)] = WINDOWS[dataset][fold][i]

    return windows


def select_test_windows(WINDOWS): # not used: test data is exported in ExportTestData.ipynb
    windows = {dataset: {} for dataset in list(WINDOWS.keys())}

    for dataset, folds_and_buildings in WINDOWS.items():
        for fold, buildings in folds_and_buildings.items():
            if fold in ['unseen_appliances']:
                windows[dataset][fold] = {}
                for i in buildings:
                    windows[dataset][fold][int(i)] = WINDOWS[dataset][fold][i]

    return windows


def _get_good_sections(df, sample_period):
    """
    Code copied from nilmtk[1]/nilmtk/stats/goodsections.py
    
    [1] https://github.com/nilmtk/nilmtk/
    """
    index = df.dropna().sort_index().index
    df_time_end = df.index[-1] + pd.Timedelta(seconds=sample_period)
    del df

    if len(index) < 2:
        return []

    timedeltas_sec = timedelta64_to_secs(np.diff(index.values))
    timedeltas_check = timedeltas_sec <= sample_period

    # Memory management
    del timedeltas_sec
    gc.collect()

    timedeltas_check = np.concatenate(
        [[False],
         timedeltas_check])
    transitions = np.diff(timedeltas_check.astype(np.int))

    # Memory management
    last_timedeltas_check = timedeltas_check[-1]
    del timedeltas_check
    gc.collect()

    good_sect_starts = list(index[:-1][transitions ==  1])
    good_sect_ends   = list(index[:-1][transitions == -1])

    # Memory management
    last_index = index[-1]
    del index
    gc.collect()

    # Work out if this chunk ends with an open ended good section
    if len(good_sect_ends) == 0:
        ends_with_open_ended_good_section = (
            len(good_sect_starts) > 0)
    elif len(good_sect_starts) > 0:
        # We have good_sect_ends and good_sect_starts
        ends_with_open_ended_good_section = (
            good_sect_ends[-1] < good_sect_starts[-1])
    else:
        # We have good_sect_ends but no good_sect_starts
        ends_with_open_ended_good_section = False

    if ends_with_open_ended_good_section:
        good_sect_ends += [df_time_end]

    assert len(good_sect_starts) == len(good_sect_ends)

    sections = [TimeFrame(start, end)
                for start, end in zip(good_sect_starts, good_sect_ends)
                if not (start == end and start is not None)]

    # Memory management
    del good_sect_starts
    del good_sect_ends
    gc.collect()

    return sections


def get_effective_good_sections(mains_meter, max_sample_period=120, min_section_length=3600):
    """
    Estimates the time frames with valid data by changes in the measurement value.
    This method does not trust the validity of values merely by it's existence.
    """
    mainsdata = mains_meter.power_series_all_data(sample_period=6, resample_kwargs={'fill_method': None, 'how': 'mean'}).dropna()
    mainsdatadiff = mainsdata.diff()
    mainsdatachanges = mainsdata[mainsdatadiff != 0.0]
    good_sections = _get_good_sections(mainsdatachanges, sample_period=max_sample_period)
    del mainsdatadiff
    del mainsdatachanges
    del mainsdata
    gc.collect()
    return [gs for gs in good_sections if gs.timedelta.total_seconds() >= min_section_length]


def load_data_from_nilmtk_datasets(windows, dataset_paths, appliances, target_appliance_name, sample_period):
    data = {}
    data_good_sections = {}

    logger.info("Loading NILMTK data...")

    for dataset_name, folds in windows.items():
        # Load dataset
        dataset = nilmtk.DataSet(dataset_paths[dataset_name])

        for fold, buildings_and_windows in folds.items():
            for building_i, windows_for_building in buildings_and_windows.items():
                dataset.set_window(None, None)
                elec = dataset.buildings[building_i].elec

                building_name = (
                    dataset.metadata['name'] +
                    '_building_{}'.format(building_i))
                logger.info(
                    "Loading data for {}...".format(building_name))
                mains_meter = elec.mains()
                good_sections = get_effective_good_sections(mains_meter)

                appliance_aliases = appliances[dataset_name][target_appliance_name]
                appliance_meters = []
                for meter in elec.meters:
                    if meter.is_site_meter():
                        continue

                    if len(meter.appliances) == 1:
                        appliancetype = meter.appliances[0].type['type']
                        if appliancetype in appliance_aliases:
                            appliance_meters.append(meter)
                    else:
                        append_meter = False
                        for a in meter.appliances:
                            if a.type['type'] in appliance_aliases:
                                append_meter = True
                        if append_meter:
                            appliance_meters.append(meter)
                            print(meter.appliances)

                if not appliance_meters:
                    logger.info(
                        "No {} found in {}".format(target_appliance_name, building_name))
                    continue

                if len(appliance_meters) > 1:
                    appliance_metergroup = nilmtk.MeterGroup(meters=appliance_meters)
                else:
                    appliance_metergroup = appliance_meters[0]
                data_good_sections.setdefault(fold, {})[building_name] = good_sections

                def load_data(meter):
                    df = meter.power_series_all_data(
                        sample_period=sample_period
                        )
                    if df is not None:
                        return df.astype(np.float32).dropna()
                    else:
                        return None

                dfs = []
                for window in windows_for_building:
                    if dataset_name == "ECO":
                        dataset.store.window = TimeFrame(start=window[0], end=window[1], tz='GMT')
                    else:
                        if window is None:
                            ipdb.set_trace() # Something has gone wrong...see what happend!
                        dataset.set_window(*window) # does not work for ECO
                    #ipdb.set_trace()
                    mains_data = load_data(mains_meter)
                    appliance_data = load_data(appliance_metergroup)
                    if (mains_data is None) or (appliance_data is None):
                        continue
                    df = pd.DataFrame(
                        {'mains': mains_data, 'target': appliance_data},
                        dtype=np.float32).dropna()
                    del mains_data
                    del appliance_data
                    if not df.empty:
                        dfs.append(df)

                df = pd.concat(dfs, axis=0)
                dfs = []
                for gs in good_sections:
                    dfslice = gs.slice(df)
                    if not dfslice.empty:
                        dfs.append(dfslice)
                df = pd.concat(dfs, axis=0)

                if not df.empty:
                    data.setdefault(fold, {})[building_name] = df

                logger.info(
                    "Loaded data from building {} for fold {}"
                    " from {} to {}."
                    .format(building_name, fold, df.index[0], df.index[-1]))

        dataset.store.close()

    logger.info("Done loading NILMTK data.")
    return data, data_good_sections


def load_nilmtk_activations( dataset_paths,
                             target_appliance_name,
                             appliance_names,
                             on_power_threshold,
                             min_on_duration,
                             min_off_duration,
                             sample_period,
                             windows,
                             sanity_check=1 ):
    """
    Parameters
    ----------
    windows : dict
        Structure example:
        {
            'UKDALE': {
                'train': {<building_i>: <window>},
                'unseen_activations_of_seen_appliances': {<building_i>: <window>},
                'unseen_appliances': {<building_i>: <window>}
            }
        }

    Returns
    -------
    all_activations : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <appliance>: {
                 <building_name>: [<activations>]
        }}}
        Each activation is a pd.Series with DatetimeIndex and the following
        metadata attributes: building, appliance, fold.
    """
    logger.info("Loading NILMTK activations...")

    if sanity_check:
        # Sanity check
        for dataset in windows:
            check_windows(windows[dataset])
    
    all_activations = {}
    for dataset_name, folds in windows.items():
        # Load dataset
        dataset = nilmtk.DataSet(dataset_paths[dataset_name])
        appliance_aliases = appliance_names[dataset_name][target_appliance_name]
        
        for fold, buildings_and_windows in folds.items():
            logger.info(
                "Loading activations for fold {}.....".format(fold))         
            for building_i, windows_for_building in buildings_and_windows.items():
                #dataset.set_window(*window)
                elec = dataset.buildings[building_i].elec
                building_name = (
                    dataset.metadata['name'] + '_building_{}'.format(building_i))
                
                appliance_meters = []
                for meter in elec.meters:
                    if meter.is_site_meter():
                        continue

                    append_meter = False
                    for a in meter.appliances:
                        if a.type['type'] in appliance_aliases:
                            append_meter = True
                    if append_meter:
                        appliance_meters.append(meter)
                        print(meter.appliances)

                if not appliance_meters:
                    logger.info(
                        "No {} found in {}".format(target_appliance_name, building_name))
                    continue

                #if appliance_meters:
                if len(appliance_meters) > 1:
                    meter = nilmtk.MeterGroup(meters=appliance_meters)
                else:
                    meter = appliance_meters[0]
                logger.info(
                    "Loading {} for {}...".format(target_appliance_name, building_name))

                meter_activations = []
                for window in windows_for_building:
                    if dataset_name == "ECO":
                        dataset.store.window = TimeFrame(start=window[0], end=window[1], tz='GMT')
                    else:
                        dataset.set_window(*window) # does not work for ECO
                    # Get activations_for_fold and process them
                    meter_activations_for_building = meter.get_activations(
                        sample_period=sample_period,
                        min_off_duration=min_off_duration,
                        min_on_duration=min_on_duration,
                        on_power_threshold=on_power_threshold,
                        resample_kwargs={'fill_method': 'ffill', 'how': 'mean', 'limit': 20})
                    #meter_activations_for_building = [activation.astype(np.float32)
                    #                     for activation in meter_activations_for_building]
                    meter_activations.extend(meter_activations_for_building)

                # Attach metadata
                #for activation in meter_activations:
                #    activation._metadata = copy(activation._metadata)
                #    activation._metadata.extend(
                #        ["building", "appliance", "fold"])
                #    activation.building = building_name
                #    activation.appliance = appliance
                #    activation.fold = fold

                # Save
                if meter_activations:
                    all_activations.setdefault(
                        fold, {}).setdefault(
                        target_appliance_name, {})[building_name] = meter_activations
                logger.info(
                    "Loaded {} {} activations from {}."
                    .format(len(meter_activations), target_appliance_name, building_name))

        dataset.store.close()
        
    logger.info("Done loading NILMTK activations.")
    return all_activations


@dataset.capture
def load_nilmtk_data( DATASET_PATHS,
                      NILMTK_EXPORT_PATH,
                      EVALUATION_DATA_PATH,
                      SAMPLE_PERIOD,
                      WINDOWS,
                      TARGET_APPLIANCE,
                      APPLIANCE_ALIASES ):
    NILMTK_EXPORT_FILENAME = os.path.join(
        NILMTK_EXPORT_PATH,
        "{}_sample_period_{}s.pkl".format(
            TARGET_APPLIANCE.replace(' ', '_'), SAMPLE_PERIOD))


    # not used: test data is exported in ExportTestData.ipynb
    #EVALUATION_DATA_FILENAME = os.path.join(
    #    EVALUATION_DATA_PATH,
    #    "{}_sample_period_{}s.pkl".format(
    #        TARGET_APPLIANCE.replace(' ', '_'), SAMPLE_PERIOD))

    if os.path.exists(NILMTK_EXPORT_FILENAME):
        print("Loading {}...".format(NILMTK_EXPORT_FILENAME))
        with open(NILMTK_EXPORT_FILENAME, 'rb') as file:
            nilmtk_data = pickle.load( file )
    else:
        # not used: test data is exported in ExportTestData.ipynb
        #testdata, _ = load_data_from_nilmtk_datasets(
        #    select_test_windows(WINDOWS),
        #    DATASET_PATHS,
        #    APPLIANCE_ALIASES,
        #    TARGET_APPLIANCE,
        #    SAMPLE_PERIOD )
        #os.makedirs(EVALUATION_DATA_PATH, exist_ok=True)
        #with open(EVALUATION_DATA_FILENAME, 'wb') as file:
        #    pickle.dump( testdata, file )
        #
        data, data_good_sections = load_data_from_nilmtk_datasets(
            select_train_windows(WINDOWS),
            DATASET_PATHS,
            APPLIANCE_ALIASES,
            TARGET_APPLIANCE,
            SAMPLE_PERIOD )
        nilmtk_data = {
            'target_appliance' : TARGET_APPLIANCE,
            'appliance_aliases' : APPLIANCE_ALIASES,
            'sample_period': SAMPLE_PERIOD,
            'windows' : select_train_windows(WINDOWS),
            'dataset_paths' : DATASET_PATHS,
            'data' : data,
            'good_sections' : data_good_sections
        }
        os.makedirs(NILMTK_EXPORT_PATH, exist_ok=True)
        with open(NILMTK_EXPORT_FILENAME, 'wb') as file:
            pickle.dump( nilmtk_data, file )

    return nilmtk_data


@dataset.capture
def load_activity_data( ACTIVITY_DATA_PATH,
                        TARGET_APPLIANCE,
                        APPLIANCE_ALIASES,
                        SAMPLE_PERIOD,
                        WINDOWS,
                        DATASET_PATHS,
                        ON_POWER_THRESHOLD,
                        MIN_ON_DURATION,
                        MIN_OFF_DURATION ):

    ACTIVITY_DATA_FILENAME = os.path.join(
        ACTIVITY_DATA_PATH,
        "{}_sample_period_{}s.pkl".format(
            TARGET_APPLIANCE.replace(' ', '_'), SAMPLE_PERIOD))
    if os.path.exists(ACTIVITY_DATA_FILENAME):
        print("Loading {}...".format(ACTIVITY_DATA_FILENAME))
        with open(ACTIVITY_DATA_FILENAME, 'rb') as file:
            real_aggregate_activity_data = pickle.load(file)
    else:
        nilmtk_data = load_nilmtk_data()
        nilmtk_activations = load_nilmtk_activations(
            DATASET_PATHS,
            TARGET_APPLIANCE,
            APPLIANCE_ALIASES,
            ON_POWER_THRESHOLD,
            MIN_ON_DURATION,
            MIN_OFF_DURATION,
            SAMPLE_PERIOD,
            select_train_windows(WINDOWS),
            sanity_check=0)

        real_aggregate_activity_data = RealAggregateActivityData(
            TARGET_APPLIANCE,
            nilmtk_activations,
            nilmtk_data,
            SAMPLE_PERIOD)

        os.makedirs(ACTIVITY_DATA_PATH, exist_ok=True)
        with open(ACTIVITY_DATA_FILENAME, 'wb') as file:
            pickle.dump( real_aggregate_activity_data, file )

    return real_aggregate_activity_data


@dataset.capture
def get_sources(training_source_names,
                validation_source_names,
                seq_length,
                sources_seed,
                VAMPIRE_POWER,
                target_inclusion_prob=0.5,
                remove_vampire_power=False,
                stride=1,
                validation_stride=128,
                validation_seq_length=None):
    """
    possible source names: ["BalancedActivityRealAggregateSource", "BalancedActivityAugmentedAggregateSource"]
    """
    training_sources = OrderedDict()
    validation_sources = OrderedDict()
    real_aggregate_activity_data = load_activity_data()
    if validation_seq_length is None:
        validation_seq_length = seq_length

    if not remove_vampire_power:
        VAMPIRE_POWER = []

    ### TRAINING

    if "BalancedActivityRealAggregateSource" in training_source_names:
        bas = BalancedActivityRealAggregateSource(
            activity_data=real_aggregate_activity_data,
            seq_length=seq_length,
            target_inclusion_prob=target_inclusion_prob,
            allow_incomplete_target=True,
            vampire_power_per_building=VAMPIRE_POWER,
            rng_seed=sources_seed
        )
        training_sources["BalancedActivityRealAggregateSource"] = bas

    if "BalancedActivityAugmentedAggregateSource" in training_source_names:
        aas = BalancedActivityAugmentedAggregateSource(
            activity_data=real_aggregate_activity_data,
            seq_length=seq_length,
            target_inclusion_prob=target_inclusion_prob,
            allow_incomplete_target=True,
            vampire_power_per_building=VAMPIRE_POWER,
            rng_seed=sources_seed
        )
        training_sources["BalancedActivityAugmentedAggregateSource"] = aas

    if "RandomizedSequentialSource" in training_source_names:
        rss = RandomizedSequentialSource(
            activity_data=real_aggregate_activity_data,
            seq_length=seq_length,
            stride=stride,
            vampire_power_per_building=VAMPIRE_POWER,
            rng_seed=sources_seed
        )
        training_sources["RandomizedSequentialSource"] = rss

    if "BalancedBuildingRealAggregateSource" in training_source_names:
        bbs = BalancedBuildingRealAggregateSource(
            activity_data=real_aggregate_activity_data,
            seq_length=seq_length,
            stride=stride,
            vampire_power_per_building=VAMPIRE_POWER,
            rng_seed=sources_seed
        )
        training_sources["BalancedBuildingRealAggregateSource"] = bbs

    ### VALIDATION

    if "RandomizedSequentialSource" in validation_source_names:
        rss = RandomizedSequentialSource(
            activity_data=real_aggregate_activity_data,
            seq_length=validation_seq_length,
            stride=validation_stride,
            vampire_power_per_building=VAMPIRE_POWER,
            rng_seed=sources_seed
        )
        validation_sources["RandomizedSequentialSource"] = rss

    return training_sources, validation_sources


@dataset.command
def convert():
    load_activity_data()
    return True


@standalone.main
def standalone_main():
    load_activity_data()
    return True    

if __name__ == '__main__':
    standalone.run_commandline()
