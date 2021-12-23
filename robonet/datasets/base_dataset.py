import os
import tensorflow as tf
#from tensorflow.contrib.training import HParams
from hparams import HParams
import glob
import copy
from .util.metadata_helper import load_metadata, MetaDataContainer
import random
import numpy as np


class BaseVideoDataset(object):
    def __init__(self, batch_size, dataset_files_or_metadata, hparams=dict()):
        assert isinstance(batch_size, int), "batch_size must be an integer"
        self._batch_size = batch_size
        if isinstance(dataset_files_or_metadata, str):
            self._metadata = [load_metadata(dataset_files_or_metadata)]
        elif isinstance(dataset_files_or_metadata, MetaDataContainer):
            self._metadata = [dataset_files_or_metadata]
        elif isinstance(dataset_files_or_metadata, (list, tuple)):
            self._metadata = []
            for d in dataset_files_or_metadata:
                assert isinstance(d, (str, MetaDataContainer)), "potential dataset must be folder containing files or meta-data instance"
                if isinstance(d, str):
                    self._metadata.append(load_metadata(d))
                else:
                    self._metadata.append(d)

        # initialize hparams, and update with user specified ones
        self._hparams = self._get_default_hparams()
        for key in self._hparams.keys():
            setattr(self._hparams, key, self._hparams[key])
        for key in hparams:
            assert key in self._hparams
            setattr(self._hparams, key, hparams[key])

        self._init_rng()
        #initialize dataset
        self._num_ex_per_epoch = self._init_dataset()
        print('loaded {} train files'.format(self._num_ex_per_epoch))

    def _init_dataset(self):
        return 0

    def _init_rng(self):
        # if RNG is not supplied then initialize new RNG
        self._random_generator = {}
        
        seeds = [None for _ in range(len(self.modes) + 1)]
        if self._hparams.RNG:
            seeds = [i + self._hparams.RNG for i in range(len(seeds))]
        
        for k, seed in zip(self.modes + ['base'], seeds):
            if k == 'train' and self._hparams.use_random_train_seed:
                seed = None
            self._random_generator[k] = random.Random(seed)
        self._np_rng = np.random.RandomState(self._random_generator['base'].getrandbits(32))
    
    def _get(self, key, mode):
        raise NotImplementedError

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,
            'use_random_train_seed': False,
        }
        return HParams(**default_dict)
    
    def get(self, key, mode='train'):
        if mode not in self.modes:
            raise ValueError('Mode {} not valid! Dataset has following modes: {}'.format(mode, self.modes))
        return self._get(key, mode)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)
    
    def __contains__(self, item):
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def hparams(self):
        return copy.deepcopy(self._hparams)

    @property
    def num_examples_per_epoch(self):
        return self._num_ex_per_epoch
    
    @property
    def modes(self):
        return ['train', 'val', 'test']

    @property
    def primary_mode(self):
        return 'train'

    def build_feed_dict(self, mode):
        return {}
