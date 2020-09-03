from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np

import os, pickle

from tqdm import tqdm
from collections import OrderedDict

## PYTORCH
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("AGG")
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.datathread import DataThread
from neuralnilm.data.dataprocess import DataProcess
from neuralnilm.data.processing import DivideBy, IndependentlyCenter, Transpose, DownSample
from neuralnilm.consts import DATA_FOLD_NAMES

from disaggregator import Disaggregator
from metrics import MetricsAccumulator
from parallelsource import MultiprocessActivationsSource
from utils import SW_add_scalars2

from sacred import Experiment
from sacred.observers import FileStorageObserver

from dataset_final_URED_ import dataset, load_activity_data, get_sources

ex = Experiment(ingredients=[dataset])

#from sacred.utils import apply_backspaces_and_linefeeds
#ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def base_experiment_config(dataset):
    INPUT_MEAN = 500.0
    INPUT_STD = 700.0

    TARGET_APPLIANCE = dataset["TARGET_APPLIANCE"]
    SOURCE_TYPES = ["BalancedActivityRealAggregateSource", "BalancedActivityAugmentedAggregateSource"]
    VALIDATION_SOURCE_TYPES = ["RandomizedSequentialSource"]

    if TARGET_APPLIANCE == 'television':
        MAX_TARGET_POWER = 1200
        SEQ_LENGTH = 1024 + 512
    elif TARGET_APPLIANCE == 'microwave':
        MAX_TARGET_POWER = 3000
        SEQ_LENGTH = 288
    elif TARGET_APPLIANCE == 'tumble dryer':
        MAX_TARGET_POWER = 2500
        SEQ_LENGTH = 1024
    elif TARGET_APPLIANCE == 'fridge':
        SOURCE_TYPES = ["BalancedBuildingRealAggregateSource"]
        MAX_TARGET_POWER = 300
        SEQ_LENGTH = 512
    elif TARGET_APPLIANCE == 'kettle':
        MAX_TARGET_POWER = 3100
        SEQ_LENGTH = 128
    elif TARGET_APPLIANCE == 'washing machine':
        MAX_TARGET_POWER = 2500
        SEQ_LENGTH = 1024
    elif TARGET_APPLIANCE == 'dish washer':
        MAX_TARGET_POWER = 2500
        SEQ_LENGTH = 1024 + 512

    TRAINING_SEED = 42
    VERBOSE_TRAINING = True
    DOWNSAMPLE_FACTOR = 0

    DESCRIPTION = """ Re-Impl. of JK's dAE in PyTorch
    """

    NUM_SEQ_PER_BATCH = 64
    EPOCHS = 100
    STEPS_PER_EPOCH = 1000

    LEARNING_RATE = 1e-1
    NUM_BATCHES_FOR_VALIDATION = 64
    USE_CUDA = True
    CHECKPOINT_BEST_MSE = False
    CHECKPOINTING_EVERY_N_EPOCHS = None
    TEST_DISAGGREGATE_EVERY_N_EPOCHS = 1


@ex.capture
def get_validation_batches(data_pipeline, VALIDATION_SOURCE_TYPES, NUM_BATCHES_FOR_VALIDATION):
    shortname_sources = {
        "BalancedActivityRealAggregateSource": "bas",
        "BalancedActivityAugmentedAggregateSource": "aas",
        "RandomizedSequentialSource": "rss",
        "BalancedBuildingRealAggregateSource": "bbs"
    }
    shortname_folds = { # listed are only validation folds
        "unseen_activations_of_seen_appliances": "unseen_activations"
    }

    validation_batches = {}

    for source in VALIDATION_SOURCE_TYPES:
        for fold in shortname_folds:
            fold_batches = []
            try:
                batch = data_pipeline.get_batch(fold=fold, source_id=source, reset_iterator=True, validation=False)
                fold_batches.append(batch)
            except KeyError:
                print("For fold `{}` no validation data available".format(fold))
                continue

            for i in range(1,NUM_BATCHES_FOR_VALIDATION):
                batch = data_pipeline.get_batch(fold=fold, source_id=source, validation=False)
                if batch is None: # Here we reach the end of the validation data for the current fold.
                    break
                else:
                    fold_batches.append(batch)

            if i == NUM_BATCHES_FOR_VALIDATION-1:
                print("WARNING: Validation data may not be fully covered")

            validation_batches[(shortname_folds[fold],shortname_sources[source])] = fold_batches
    return validation_batches


def disag_seq2seq(model, mains, target, input_processing, target_processing, max_target_power, n_seq_per_batch, seq_length, target_seq_length, building_i, stride, USE_CUDA, return_sequences):
    def apply_inverse_processing(batch, processing_steps):
        reversed_processing_steps = processing_steps[::-1]
        for step in reversed_processing_steps:
            batch = step.inverse(batch)

        return batch

    def apply_processing(batch, processing_steps):
        for step in processing_steps:
            batch = step(batch)

        return batch

    def mains_to_batches(mains, n_seq_per_batch, seq_length, processing_steps, stride=1):
        assert mains.ndim == 1
        n_mains_samples = len(mains)
        input_shape = (n_seq_per_batch, seq_length, 1)

        # Divide mains data into batches
        n_batches = (n_mains_samples / stride) / n_seq_per_batch
        n_batches = np.ceil(n_batches).astype(int)
        batches = []
        for batch_i in range(n_batches):
            batch = np.zeros(input_shape, dtype=np.float32)
            batch_start = batch_i * n_seq_per_batch * stride
            for seq_i in range(n_seq_per_batch):
                mains_start_i = batch_start + (seq_i * stride)
                mains_end_i = mains_start_i + seq_length
                seq = mains[mains_start_i:mains_end_i]
                batch[seq_i, :len(seq), 0] = seq
            processed_batch = apply_processing(batch, processing_steps)
            batches.append(processed_batch)

        return batches


    if stride is None:
        stride = seq_length

    batches = mains_to_batches(mains, n_seq_per_batch, seq_length, input_processing, stride)
    estimates = np.zeros(len(mains), dtype=np.float32)
    offset = (seq_length - target_seq_length) // 2

    if return_sequences:
        # `estimate_windows` is array with shape [#sliding_windows_in_mains x lenth_of_window]
        #.   it stores disag results for all sliding windows seperately
        # note: beware of padding. also last batch may be not be filled entirely
        #.      therefore have some extra margin
        estimate_windows = np.zeros((len(batches)*n_seq_per_batch, target_seq_length), dtype=np.float32)

    # Iterate over each batch
    window_i = 0
    for batch_i, net_input in enumerate(batches):
        input = torch.from_numpy(net_input.astype(np.float32))
        if USE_CUDA:
            input = input.cuda()
        with torch.no_grad():
            inputv = Variable(input)
            output = model(inputv)
            net_output = output.cpu().data.numpy()
        net_output = apply_inverse_processing(net_output, target_processing)
        batch_start = (batch_i * n_seq_per_batch * stride) + offset
        for seq_i in range(n_seq_per_batch):
            start_i = batch_start + (seq_i * stride)
            end_i = start_i + target_seq_length
            n = len(estimates[start_i:end_i])
            # The net output is not necessarily the same length
            # as the mains (because mains might not fit exactly into
            # the number of batches required)
            estimates[start_i:end_i] += net_output[seq_i, :n, 0]
            if return_sequences:
                estimate_windows[window_i, :] = net_output[seq_i, :, 0]

            window_i += 1

    n_overlaps = target_seq_length / stride
    estimates /= n_overlaps
    estimates[estimates < 0] = 0

    if return_sequences:
        return estimates, dict(sequences=estimate_windows)
    else:
        return estimates


class _JKsDenoisingAutoEncoderOriginal(nn.Module):
    def __init__(self, SEQ_LENGTH):
        super(_JKsDenoisingAutoEncoderOriginal, self).__init__()

        self.seq_length = SEQ_LENGTH
        self.nc  = 1 # number of channels
        self.nef = 8 # number of filters
        nrepr = 128 # dimension of internal representation
        self.fs  = 4 # filter size

        self.pad1 = nn.ConstantPad1d((1, 2), 0)
        self.conv1 = nn.Conv1d(self.nc, self.nef, self.fs, stride=1, padding=0)

        self.n_dense_units = self.nef*self.seq_length
        self.dense1 = nn.Sequential(
            nn.Linear(self.n_dense_units, self.n_dense_units),
            nn.ReLU(True),

            nn.Linear(self.n_dense_units, nrepr),
            nn.ReLU(True),

            nn.Linear(nrepr, self.n_dense_units),
            nn.ReLU(True)
        )

        self.pad2 = nn.ConstantPad1d((1, 2), 0)
        self.conv2 = nn.Conv1d(self.nef, self.nc, self.fs, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, input):
        #TODO: Die Anzahl der veruegbaren GPUs beruecksichtigen.
        x = self.pad1(input)
        x = self.conv1(x)
        x = self.dense1(x.view(-1, self.n_dense_units))
        x = self.pad2(x.view(-1, self.nef, self.seq_length))
        x = self.conv2(x)
        return x

_Net = _JKsDenoisingAutoEncoderOriginal


def load_disaggregator(EVALUATION_DATA_PATH, MODEL_PATH, config=None, USE_CUDA=True):
    """
        Helper function for the disaggregator script
    """

    if config is None:
        config = os.path.dirname(MODEL_PATH)

    if type(config) == str:
        try:
            import jsonpickle
            with open(os.path.join(config, 'config.json'), 'r') as configfile:
                config = jsonpickle.decode(configfile.read())
        except:
            return None

    assert(type(config) == dict)

    dataset = config['dataset']
    SEQ_LENGTH =            config['SEQ_LENGTH']
    TARGET_APPLIANCE =      dataset['TARGET_APPLIANCE']
    ON_POWER_THRESHOLD =    dataset['ON_POWER_THRESHOLD']
    MAX_TARGET_POWER =      config['MAX_TARGET_POWER']
    NUM_SEQ_PER_BATCH =     config['NUM_SEQ_PER_BATCH']
    INPUT_STD =             config['INPUT_STD']
    INPUT_MEAN =            config['INPUT_MEAN']
    DOWNSAMPLE_FACTOR =     config['DOWNSAMPLE_FACTOR']

    net = _Net(SEQ_LENGTH)

    input_processing_steps = [DivideBy(INPUT_STD), IndependentlyCenter(), Transpose((0, 2, 1))]
    target_processing_steps = [DivideBy(MAX_TARGET_POWER), Transpose((0, 2, 1))]

    if MODEL_PATH.endswith("/"):
        MODEL_PATH = MODEL_PATH + 'net_step_{:06d}.pth.tar'.format(config['EPOCHS']*config['STEPS_PER_EPOCH'])

    if USE_CUDA:
        training_state = torch.load(MODEL_PATH)
    else:
        training_state = torch.load(MODEL_PATH, map_location='cpu')

    if MODEL_PATH.endswith("tar"):
        model = training_state['model']
    else:
        model = training_state

    net.load_state_dict(model)
    if USE_CUDA:
        net.cuda()

    return Disaggregator(
        EVALUATION_DATA_PATH=EVALUATION_DATA_PATH,
        TARGET_APPLIANCE = TARGET_APPLIANCE,
        ON_POWER_THRESHOLD = ON_POWER_THRESHOLD,
        MAX_TARGET_POWER = MAX_TARGET_POWER,
        pad_mains = True,
        pad_appliance = False,
        disagg_func = disag_seq2seq,
        downsample_factor = DOWNSAMPLE_FACTOR,
        disagg_kwargs = dict(
            USE_CUDA=USE_CUDA,
            model = net,
            input_processing=input_processing_steps,
            target_processing=target_processing_steps,
            n_seq_per_batch = NUM_SEQ_PER_BATCH,
            seq_length = SEQ_LENGTH,
            target_seq_length = SEQ_LENGTH,
            stride = 16
        )
    ), training_state


@ex.automain
def run_experiment(dataset,
                   INPUT_STD,
                   SOURCE_TYPES,
                   VALIDATION_SOURCE_TYPES,
                   DOWNSAMPLE_FACTOR,
                   SEQ_LENGTH,
                   MAX_TARGET_POWER,
                   TARGET_APPLIANCE,
                   TRAINING_SEED,
                   VERBOSE_TRAINING,
                   LEARNING_RATE,
                   NUM_SEQ_PER_BATCH,
                   EPOCHS,
                   STEPS_PER_EPOCH,
                   USE_CUDA,
                   CHECKPOINT_BEST_MSE,
                   CHECKPOINTING_EVERY_N_EPOCHS,
                   TEST_DISAGGREGATE_EVERY_N_EPOCHS,
                   _run):

    torch.manual_seed(TRAINING_SEED)

    OUTPUT_FOLDER = os.path.join(ex.get_experiment_info()['name'],"output")
    for observer in _run.observers:
        if type(observer) is FileStorageObserver:
            OUTPUT_FOLDER = os.path.join(observer.basedir, str(_run._id))
            VERBOSE_TRAINING = 0
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    writer = SummaryWriter(log_dir=OUTPUT_FOLDER)

    # From dataset Ingredient
    TRAIN_BUILDINGS = dataset["TRAIN_BUILDINGS"]
    ON_POWER_THRESHOLD = dataset["ON_POWER_THRESHOLD"]

    ##############################################################################################
    #PREPARE DATASET (DATALOADERs)
    ##############################################################################################
    running_data_processes = [] # stop these at the end
    sources, validation_sources = get_sources(
        training_source_names=SOURCE_TYPES,
        validation_source_names=VALIDATION_SOURCE_TYPES,
        seq_length=SEQ_LENGTH,
        sources_seed=TRAINING_SEED,
        validation_stride=128 )

    if DOWNSAMPLE_FACTOR > 1:
        downsample_rng = np.random.RandomState(TRAINING_SEED)
        input_processing_steps = [DownSample(DOWNSAMPLE_FACTOR, downsample_rng)]
    else:
        input_processing_steps = []
    input_processing_steps += [DivideBy(INPUT_STD), IndependentlyCenter(), Transpose((0, 2, 1))]
    target_processing_steps = [DivideBy(MAX_TARGET_POWER), Transpose((0, 2, 1))]

    validation_pipeline = DataPipeline(
        sources=validation_sources,
        num_seq_per_batch=NUM_SEQ_PER_BATCH,
        input_processing=input_processing_steps,
        target_processing=target_processing_steps
    )
    validation_batches = get_validation_batches(validation_pipeline)
    print("appliance {} has {} validation batches".format(
        TARGET_APPLIANCE,
        sum([len(v) for k, v in validation_batches.items()]) ))

    data_pipeline = DataPipeline(
        sources=sources,
        num_seq_per_batch=NUM_SEQ_PER_BATCH,
        input_processing=input_processing_steps,
        target_processing=target_processing_steps
    )
    data_thread = DataProcess(data_pipeline)
    data_thread.start()
    running_data_processes.append(data_thread)

    net = _Net(SEQ_LENGTH)
    print(net)

    metrics_accu = MetricsAccumulator(
        on_power_threshold=ON_POWER_THRESHOLD,
        max_power=MAX_TARGET_POWER)

    # note: MSE - Mean Squared Error
    criterion = torch.nn.MSELoss()

    stop_training = False
    best_mse = None

    # PREPARE DISAGGREGATOR
    if TEST_DISAGGREGATE_EVERY_N_EPOCHS is not None:
	    test_disaggregator = Disaggregator(
	        EVALUATION_DATA_PATH = dataset['EVALUATION_DATA_PATH'],
	        TARGET_APPLIANCE = TARGET_APPLIANCE,
	        ON_POWER_THRESHOLD = ON_POWER_THRESHOLD,
	        MAX_TARGET_POWER = MAX_TARGET_POWER,
	        pad_mains = True,
	        pad_appliance = False,
	        disagg_func = disag_seq2seq,
	        downsample_factor = DOWNSAMPLE_FACTOR,
	        disagg_kwargs = dict(
	            model = net,
	            input_processing=input_processing_steps,
	            target_processing=target_processing_steps,
	            n_seq_per_batch = NUM_SEQ_PER_BATCH,
	            seq_length = SEQ_LENGTH,
	            target_seq_length = SEQ_LENGTH,
	            USE_CUDA=USE_CUDA,
	            stride = 16
	        )
	    )

    # PREPARE TENSORS, WHICH WILL BE FED USED DURING TRAINING AND VALIDATION
    input = torch.FloatTensor(NUM_SEQ_PER_BATCH, 1, SEQ_LENGTH)
    target = torch.FloatTensor(NUM_SEQ_PER_BATCH, 1, SEQ_LENGTH)

    if USE_CUDA:
        # note: push to GPU
        net.cuda()
        criterion.cuda()
        input, target = input.cuda(), target.cuda()

    # setup optimizer.  TODO: Should we use 'Adam' for disaggregator?
    #optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(net.parameters(), momentum=0.9, nesterov=True, lr=LEARNING_RATE)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    history = {}
    csvpath = os.path.join(OUTPUT_FOLDER, "history.csv")
    if os.path.exists(csvpath):
        print("Already exists: {}".format(csvpath))
        return -1

    progbar_epoch = tqdm(desc="Epoch", total=EPOCHS, unit="epoch", disable=(not VERBOSE_TRAINING))
    for epoch in range(EPOCHS):
        # TRAINING
        metrics_log = {'training':{}}
        training_loss = 0.0
        progbar = tqdm(desc="Train", total=STEPS_PER_EPOCH, leave=False, disable=(not VERBOSE_TRAINING))
        for i in range(STEPS_PER_EPOCH):
            net.zero_grad()
            batch = data_thread.get_batch()
            while batch is None:
                batch = data_thread.get_batch()
            qsize = data_thread._queue.qsize()

            aggregated_signal = torch.from_numpy(batch.after_processing.input)
            target_signal = torch.from_numpy(batch.after_processing.target)
            if USE_CUDA:
                aggregated_signal = aggregated_signal.cuda()
                target_signal = target_signal.cuda()
            input.resize_as_(aggregated_signal).copy_(aggregated_signal)
            target.resize_as_(target_signal).copy_(target_signal)
            inputv = Variable(input, requires_grad=False)
            targetv = Variable(target, requires_grad=False)
            output = net(inputv)
            loss = criterion(output, targetv)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            progbar.set_postfix(dict(
                    loss = "{:.4f}".format(loss.item()),
                    qsize = qsize
                ), refresh=False)
            progbar.update()

        metrics_log['training']['loss'] = float(training_loss/STEPS_PER_EPOCH)
        metrics_log['training']['lr'] = optimizer.param_groups[0]['lr']

        # VALIDATION
        #pr_num_thresholds = 127
        for fold in validation_batches:
            metrics_accu.reset_accumulator()
            #accumulated_pr = {}
            #for cl in ["tp", "tn", "fp", "fn"]:
            #    accumulated_pr[cl] = torch.LongTensor(pr_num_thresholds).zero_()
            for batch in validation_batches[fold]:
                aggregated_signal = torch.from_numpy(batch.after_processing.input)
                target_signal = torch.from_numpy(batch.after_processing.target)
                if USE_CUDA:
                    aggregated_signal = aggregated_signal.cuda()
                    target_signal = target_signal.cuda()
                input.resize_as_(aggregated_signal).copy_(aggregated_signal)
                target.resize_as_(target_signal).copy_(target_signal)
                with torch.no_grad():
                    inputv = Variable(input)
                    targetv = Variable(target)
                    output = net(inputv)
                    val_loss = criterion(output, targetv)
                    loss_value = val_loss.item()
                # other metrics
                pred_y = data_pipeline.apply_inverse_processing(output.cpu().data.numpy(), 'target')
                true_y = batch.before_processing.target
                metrics_accu.accumulate_metrics(true_y, pred_y, val_loss=loss_value)
                #calculate_pr_curve_torch(accumulated_pr, MAX_TARGET_POWER, true_y, pred_y, num_thresholds=pr_num_thresholds)

            for key, value in metrics_accu.finalize_metrics().items():
                metrics_log.setdefault(fold[0], {}).setdefault(key, {})[fold[1]] = value

            #precision = accumulated_pr["tp"] / (accumulated_pr["tp"] + accumulated_pr["fp"])
            #recall = accumulated_pr["tp"] / (accumulated_pr["tp"] + accumulated_pr["fn"])
            #writer.add_pr_curve_raw("pr_{}/{}".format(fold[0], fold[1]),
            #    true_positive_counts=accumulated_pr["tp"],
            #    false_positive_counts=accumulated_pr["fp"],
            #    true_negative_counts=accumulated_pr["tn"],
            #    false_negative_counts=accumulated_pr["fn"],
            #    precision=precision, recall=recall,
            #    global_step=(epoch+1)*STEPS_PER_EPOCH, num_thresholds=pr_num_thresholds)

        # LR Scheduler
        val_loss = metrics_log['unseen_activations']['val_loss']['rss']
        #val_loss = metrics_log['mean_squared_error']['unseen_activations']['rss']
        #scheduler.step(val_loss)
        scheduler.step()

        # PRINT STATS
        if not VERBOSE_TRAINING:
            print('[{:d}/{:d}] {}'.format(epoch+1, EPOCHS, metrics_log['training']))
        else:
            progbar_epoch.set_postfix(dict(loss=metrics_log['training']['loss']), refresh=False)

        progbar_epoch.update()
        progbar.close()

        # store in history / tensorboard
        for fold, metrics_for_fold in metrics_log.items():
            for metric_name, value in metrics_for_fold.items():
                if type(value) == dict:
                    SW_add_scalars2(writer, "{}/{}".format(fold, metric_name), value, (epoch+1)*STEPS_PER_EPOCH)
                    for k, v in value.items():
                        name = "{}/{}/{}".format(fold, metric_name, k)
                        history.setdefault(name, []).append(v)
                else:
                    name = "{}/{}".format(fold, metric_name)
                    writer.add_scalar(name, value, (epoch+1)*STEPS_PER_EPOCH)
                    history.setdefault(name, []).append(value)

        # CHECKPOINTING
        if CHECKPOINT_BEST_MSE:
            mse = val_loss
            if best_mse is None:
                best_mse = mse
            if best_mse > mse:
                msg = "[{:d}/{:d}] MSE improved from {:.4f} to {:.4f} (d={:f}), saving model...".format(epoch+1, EPOCHS, best_mse, mse, best_mse-mse)
                if not VERBOSE_TRAINING:
                    print(msg)
                else:
                    progbar_epoch.write(msg)
                torch.save({
                    'epoch': epoch + 1,
                    'step' : (epoch+1)*STEPS_PER_EPOCH,
                    'mse'  : mse,
                    'model': net.state_dict()}, '{}/net_best_mse.pth.tar'.format(OUTPUT_FOLDER))
                best_mse = mse

        if CHECKPOINTING_EVERY_N_EPOCHS is not None:
            if (epoch+1) % CHECKPOINTING_EVERY_N_EPOCHS == 0:
                torch.save(net.state_dict(), '{}/net_step_{:06d}.pth'.format(OUTPUT_FOLDER, (epoch+1)*STEPS_PER_EPOCH))

        if TEST_DISAGGREGATE_EVERY_N_EPOCHS is not None:
            if (epoch+1) % TEST_DISAGGREGATE_EVERY_N_EPOCHS == 0:
                scores = test_disaggregator.calculate_metrics()
                scores_by_metric = {}
                for building_i, building in scores.items():
                    for metric, value in building.items():
                        scores_by_metric.setdefault(metric, {})[building_i] = value
                for metric, building_d in scores_by_metric.items():
                    SW_add_scalars2(writer, "test_score/{}".format(metric), building_d, (epoch+1)*STEPS_PER_EPOCH)

        if stop_training:
            break

    # CHECKPOINTING at end
    torch.save({
        'epoch': epoch + 1,
        'step' : (epoch+1)*STEPS_PER_EPOCH,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        #'scheduler': scheduler.state_dict()
        # TODO: scheduler is not saved this way, scheduler.state_dict() does not exist
    }, '{}/net_step_{:06d}.pth.tar'.format(OUTPUT_FOLDER, (epoch+1)*STEPS_PER_EPOCH))

    df = pd.DataFrame(history)
    df.to_csv(csvpath)

    for p in running_data_processes:
        p.stop()
    writer.close()

    #return 42
    return metrics_log['training']['loss']
