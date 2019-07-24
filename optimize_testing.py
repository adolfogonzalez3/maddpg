'''Module for evaluating learned agents against different environments.'''
import argparse
import os
from math import ceil
from pathlib import Path
from itertools import chain
from collections import defaultdict

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import custom_envs.utils.utils_plot as utils_plot
from custom_envs.envs.multioptlrs import MultiOptLRs
from custom_envs.data import load_data
from maddpg.algorithms import MaddpgInference


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_tqdm(*args):
    tqdm.write(' '.join(str(arg) for arg in args))


def flatten_arrays(arrays):
    return list(chain.from_iterable(a.ravel().tolist() for a in arrays))


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.history.append({
            'epoch': epoch,
            'weights_mean': np.mean(flatten_arrays(self.model.get_weights())),
            **logs
        })


def run_handle(env):
    '''Run handle requests until complete.'''
    data = 0
    while data is not None:
        data = env.handle_requests()


def task(path, seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''
    Run the agent on a data set.
    '''
    sequence = load_data(data_set)
    num_of_samples = len(sequence.features)
    steps_per_epoch = ceil(num_of_samples / batch_size) if batch_size else 1
    max_batches = steps_per_epoch*total_epochs

    multi_env = MultiOptLRs(data_set=data_set, max_batches=max_batches,
                            batch_size=batch_size, max_history=25,
                            reward_version=0, observation_version=3,
                            action_version=0, version=5)
    model = MaddpgInference(multi_env.observation_space,
                            multi_env.action_space,
                            shared_policy=True)
    model.load(str(path / 'model.ckpt'))
    states = multi_env.reset()
    info_list = []
    cumulative_reward = 0
    print_tqdm('Starting...')
    for epoch_no in trange(total_epochs, leave=False):
        for step in trange(steps_per_epoch, leave=False):
            actions = model.predict(states)
            actions = {key: np.squeeze(act) for key, act in actions.items()}
            states, rewards, _, infos = multi_env.step(actions)
            cumulative_reward = cumulative_reward + rewards
            info = infos
            info['step'] = epoch_no*steps_per_epoch + step
            info['cumulative_reward'] = cumulative_reward
            info['seed'] = seed
            info['epoch'] = epoch_no
            info_list.append(info)
            if info['accuracy']:
                print_tqdm('Accuracy:', info['accuracy'],
                           'Loss:', info['loss'])
    return info_list


def task_lr(seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''Train a logistic classification model.'''
    sequence = load_data(data_set)
    features = sequence.features
    labels = sequence.labels
    batch_size = len(features) if batch_size is None else batch_size
    model = tf.keras.Sequential()
    if True:
        model.add(tf.keras.layers.Dense(
            48, input_shape=features.shape[1:],
            #kernel_initializer=tf.keras.initializers.normal(0, 1, seed=seed),
            #bias_initializer=tf.keras.initializers.normal(0, 1, seed=seed),
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            activation='relu', use_bias=True
        ))
        model.add(tf.keras.layers.Dense(
            labels.shape[-1],
            #kernel_initializer=tf.keras.initializers.normal(0, 1, seed=seed),
            #bias_initializer=tf.keras.initializers.normal(0, 1, seed=seed),
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            activation='softmax', use_bias=True
        ))
    else:
        model.add(tf.keras.layers.Dense(
            labels.shape[-1], input_shape=features.shape[1:],
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            activation='softmax', use_bias=True
        ))
    model.compile(tf.train.GradientDescentOptimizer(1e-1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callback = CustomCallback()
    model.fit(features, labels, epochs=total_epochs, verbose=0, shuffle=True,
              batch_size=batch_size, callbacks=[callback])
    return callback.history


def plot_results(axes, dataframe, groupby, label=None):
    '''Plot results on multiple axes given a dataframe.'''
    grouped = dataframe.groupby(groupby)
    mean_df = grouped.mean()
    std_df = grouped.std()
    columns = set(mean_df.columns) & set(axes.keys()) - {groupby}
    for name in columns:
        utils_plot.plot_sequence(axes[name], mean_df[name], label=label)
        utils_plot.fill_between(axes[name], mean_df[name], std_df[name],
                                alpha=0.1, label=label)


def run_multi(path, trials=10, batch_size=None, total_epochs=40,
              data_set='mnist'):
    '''Run both agent evaluation and logistic classification training.'''
    path = Path(path)
    infos = list(chain.from_iterable([task(path, i, batch_size=batch_size,
                                           total_epochs=total_epochs,
                                           data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_rl = pd.DataFrame(infos)
    dataframe_rl.to_csv(str(path / 'dataframe_rl.csv'))
    infos = list(chain.from_iterable([task_lr(i, batch_size=batch_size,
                                              total_epochs=total_epochs,
                                              data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_lc = pd.DataFrame.from_dict(infos)
    dataframe_lc.to_csv(str(path / 'dataframe_lc.csv'))
    columns = ['accuracy' if col == 'acc' else col
               for col in dataframe_lc.columns]
    dataframe_lc.columns = columns
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    pyplot_attr = {
        'title': 'Performance on {} data set'.format(data_set.upper()),
        'xlabel': 'Epoch',
    }
    columns = set(dataframe_rl.select_dtypes('number').columns) - {'epoch'}
    for column in columns:
        pyplot_attr['ylabel'] = column.capitalize()
        utils_plot.set_attributes(axes[column], pyplot_attr)

    plot_results(axes, dataframe_rl, 'epoch', 'RL Gradient Descent')
    plot_results(axes, dataframe_lc, 'epoch', 'Gradient Descent')
    for name, axis in axes.items():
        utils_plot.add_legend(axis)
        axis.figure.savefig(str(path / '{}.png'.format(name)))


def main():
    '''Evaluate a trained model against logistic regression.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights", help="The path to the model weights.",
                        type=Path)
    parser.add_argument("--trials", help="The number of trials.",
                        type=int, default=1)
    parser.add_argument("--batch_size", help="The batch size.",
                        type=int, default=32)
    parser.add_argument("--total_epochs", help="The number of epochs.",
                        type=int, default=40)
    parser.add_argument("--data_set", help="The data set to trial against.",
                        type=str, default='iris')
    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.ERROR)
    run_multi(args.model_weights, args.trials, args.batch_size,
              args.total_epochs, args.data_set)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
