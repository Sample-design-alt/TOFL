'''
find the best parameters
'''
from optuna import trial, study
import optuna
import yaml
from utils.utils import convert_to_tuner_config
from optim.pretrain import *
from utils.parse import parse_option
import os

config_path = './config/test.yaml'
configuration = yaml.safe_load(open(config_path, 'r'))


def objective(trial):
    print("Running trial #{}".format(trial.number))
    opt = parse_option()
    opt.dataset_name = 'UWaveGestureLibraryAll'
    opt.ucr_path = '../datasets'
    opt.patience = 200
    # sample the to be tuned params from the search space in the configuration file
    # and convert them to the optuna specific format
    tuner_config = convert_to_tuner_config(configuration, trial)

    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']
    exp = 'exp-tune'
    if opt.model_name == 'SemiSOP':
        model_paras = 'label{}_{}'.format(tuner_config['data_params']['label_ratio'],
                                          tuner_config['data_params']['alpha'])
    else:
        model_paras = 'label{}'.format(opt.label_ratio)

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2

    opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
        exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
        model_paras, str(1))

    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)

    x_train, y_train, x_val, y_val, x_test, y_test, tuner_config['data_params']['nb_class'], _ \
        = load_ucr2018(opt.ucr_path, opt.dataset_name)
    acc_test, acc_unlabel, epoch_max = train_SemiTime(
        x_train, y_train, x_val, y_val, x_test, y_test, opt, configuration)
    return acc_test


study = optuna.create_study(study_name='tune-UW', direction='maximize', storage='sqlite:///example.db',
                            load_if_exists=True)  # sampler=samplers.TPESampler()))
study.optimize(objective, n_trials=100)
