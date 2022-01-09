from robonet.datasets.base_dataset import BaseVideoDataset
from robonet.datasets.util.hdf5_loader import load_data, default_loader_hparams
#from tensorflow.contrib.training import HParams
from hparams import HParams
from robonet.datasets.util.dataset_utils import color_augment, split_train_val_test
from robonet.datasets.util.randaug import rand_aug, rand_crop
import numpy as np
import tensorflow as tf
import copy
import multiprocessing
import pdb


def _load_data(inputs):
    if len(inputs) == 3:
        f_name, file_metadata, hparams = inputs
        return load_data(f_name, file_metadata, hparams)
    elif len(inputs) == 4:
        f_name, file_metadata, hparams, rng = inputs
        return load_data(f_name, file_metadata, hparams, rng)
    raise ValueError


class RoboNetDataset(BaseVideoDataset):
    def __init__(self, batch_size, dataset_files_or_metadata, hparams=dict(), batch_length=30):
        source_probs = hparams.pop('source_selection_probabilities', None)
        self.batch_length = batch_length
        super(RoboNetDataset, self).__init__(batch_size, dataset_files_or_metadata, hparams)
        self._hparams.source_probs = source_probs

    def _init_dataset(self):
        if self._hparams.load_random_cam and self._hparams.same_cam_across_sub_batch:
            for s in self._metadata:
                min_ncam = min(s['ncam'])
                if sum(s['ncam'] != min_ncam):
                    print('sub-batch has data with different ncam but each same_cam_across_sub_batch=True! Could result in un-even cam sampling')
                    break
        
        # check batch_size
        assert self._batch_size % self._hparams.sub_batch_size == 0, "sub batches should evenly divide batch_size!"
        # assert np.isclose(sum(self._hparams.splits), 1) and all([0 <= i <= 1 for i in self._hparams.splits]), "splits is invalid"
        assert self._hparams.load_T >=0, "load_T should be non-negative!"

        # set output format
        output_format = [tf.uint8, tf.float32, tf.float32]        
        if self._hparams.load_annotations:
            output_format = output_format + [tf.float32]
        
        if self._hparams.ret_fnames:
            output_format = output_format + [tf.string]
        output_format = tuple(output_format)

        # smallest max step length of all dataset sources 
        min_steps = min([min(min(m.frame['img_T']), min(m.frame['state_T'])) for m in self._metadata])
        if not self._hparams.load_T:
            self._hparams.load_T = min_steps
        else:
            assert self._hparams.load_T <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams.min_T, min_steps)

        self._n_workers = min(self._batch_size, multiprocessing.cpu_count())
        if self._hparams.pool_workers:
            self._n_workers = min(self._hparams.pool_workers, multiprocessing.cpu_count())
        self._n_pool_resets = 0
        self._pool = multiprocessing.Pool(self._n_workers)

        n_train_ex = 0
        mode_sources = [[] for _ in range(len(self.modes))]
        for m_ind, metadata in enumerate(self._metadata):
            files_per_source = self._split_files(m_ind, metadata)
            assert len(files_per_source) == len(self.modes), "files should be split into {} sets (it's okay if sets are empty)".format(len(self.modes))
            for m, fps in zip(mode_sources, files_per_source):
                if len(fps):
                    self._random_generator['base'].shuffle(fps)
                    m.append((fps, metadata))                

        #self._place_holder_dict = self._get_placeholders()
        self.mode_generators = {}
        self.mode_num_files = {}
        for name, m in zip(self.modes, mode_sources):
            if m:
                rng = self._random_generator[name]
                mode_source_files, mode_source_metadata = [[t[j] for t in m] for j in range(2)]
                
                gen_func = self._wrap_generator(mode_source_files, mode_source_metadata, rng, name)
               
                dataset = tf.data.Dataset.from_generator(gen_func, output_format)
                dataset = dataset.map(self._get_dict)
                #self.mode_generators[name] = dataset
                self.mode_generators[name] = dataset.prefetch(self._hparams.buffer_size)
                self.mode_num_files[name] = sum([len(f) for f in mode_source_files])

                # dataset = dataset.map(self._get_dict)
                # dataset = dataset.prefetch(self._hparams.buffer_size)
                # self.mode_generators[name] = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return self.mode_num_files['train']

    def _split_files(self, source_number, metadata):
        if self._hparams.train_ex_per_source != [-1]:
            return split_train_val_test(metadata, train_ex=self._hparams.train_ex_per_source[source_number], rng=self._random_generator['base'])
        return split_train_val_test(metadata, splits=self._hparams.splits, rng=self._random_generator['base'])

    def _get(self, key, mode):
        if mode == self.primary_mode:
            return self._data_loader_dict[key]
        
        if key == 'images':
            return self._img_tensor
        
        return self._place_holder_dict[key]
    
    def __contains__(self, item):
        return item in self._place_holder_dict

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,             # random seed to initialize data loader rng
            'use_random_train_seed': False,          # use a random seed for training objects 
            'sub_batch_size': 1,                     # sub batch to sample from each source
            'splits': (0.9, 0.05, 0.05),             # train, val, test
            'num_epochs': None,                      # maximum number of epochs (None if iterate forever)
            'ret_fnames': False,                     # return file names of loaded hdf5 record
            'buffer_size': 100,                      # examples to prefetch
            'all_modes_max_workers': True,           # use multi-threaded workers regardless of the mode
            'load_random_cam': True,                 # load a random camera for each example
            'same_cam_across_sub_batch': False,      # same camera across sub_batches
            'pool_workers': 0,                       # number of workers for pool (if 0 uses batch_size workers)
            'color_augmentation':0.0,                # std of color augmentation (set to 0 for no augmentations)
            'train_ex_per_source': [-1],             # list of train_examples per source (set to [-1] to rely on splits only)
            'pool_timeout': 10,                      # max time to wait to get batch from pool object
            'MAX_RESETS': 10,                         # maximum number of pool resets before error is raised in main thread
            'source_probs': None,
        }
        for k, v in default_loader_hparams().items():
            default_dict[k] = v
        
        return HParams(**default_dict)

    def _wrap_generator(self, source_files, source_metadata, rng, mode):
        return lambda : self._hdf5_generator(source_files, source_metadata, rng, mode)

    def _hdf5_generator(self, sources, sources_metadata, rng, mode): 
        file_indices, source_epochs = [[0 for _ in range(len(sources))] for _ in range(2)]
        self.next_file_index = None
        #while True:
        assert len(sources) == 1
        num_batches = len(sources[0])//self._batch_size
        for _ in range(num_batches):
            file_hparams = [copy.deepcopy(self._hparams) for _ in range(self._batch_size)]
            if self._hparams.RNG:
                file_rng = [rng.getrandbits(64) for _ in range(self._batch_size)]
            else:
                file_rng = [None for _ in range(self._batch_size)]
            
            file_names, file_metadata = [], []
            b = 0
            sources_selected_thus_far = []
            while len(file_names) < self._batch_size:
                # if source_probs is set do a weighted random selection
                if self._hparams.source_probs:
                    selected_source = self._np_rng.choice(len(sources), 1, p=self._hparams.source_probs)[0]
                # if # sources <= # sub_batches then sample each source at least once per batch
                elif len(sources) <= self._batch_size // self._hparams.sub_batch_size and b // self._hparams.sub_batch_size < len(sources):
                    selected_source = b // self._hparams.sub_batch_size
                elif len(sources) > self._batch_size // self._hparams.sub_batch_size:
                    selected_source = rng.randint(0, len(sources) - 1)
                    while selected_source in sources_selected_thus_far:
                        selected_source = rng.randint(0, len(sources) - 1)
                else:
                    selected_source = rng.randint(0, len(sources) - 1)
                sources_selected_thus_far.append(selected_source)

                for sb in range(self._hparams.sub_batch_size):
                    
                    selected_file = sources[selected_source][file_indices[selected_source]]
                    file_indices[selected_source] += 1
                    selected_file_metadata = sources_metadata[selected_source].get_file_metadata(selected_file)

                    file_names.append(selected_file)
                    file_metadata.append(selected_file_metadata)
                    
                    if file_indices[selected_source] >= len(sources[selected_source]):
                        file_indices[selected_source] = 0
                        source_epochs[selected_source] += 1

                        if mode == 'train' and self._hparams.num_epochs is not None and source_epochs[selected_source] >= self._hparams.num_epochs:
                            break
                        rng.shuffle(sources[selected_source])

                b += self._hparams.sub_batch_size

            if self._hparams.load_random_cam:
                b = 0
                while b < self._batch_size:
                    if self._hparams.same_cam_across_sub_batch:
                        selected_cam = [rng.randint(0, min([file_metadata[b + sb]['ncam'] for sb in range(self._hparams.sub_batch_size)]) - 1)]
                        for sb in range(self._hparams.sub_batch_size):
                            file_hparams[b + sb].cams_to_load = selected_cam
                        b += self._hparams.sub_batch_size
                    else:
                        file_hparams[b].cams_to_load = [rng.randint(0, file_metadata[b]['ncam'] - 1)]
                        b += 1

            batch_jobs = [(fn, fm, fh, fr) for fn, fm, fh, fr in zip(file_names, file_metadata, file_hparams, file_rng)]

            try:
                batches = self._pool.map_async(_load_data, batch_jobs).get(timeout=self._hparams.pool_timeout)
            except:
                self._pool.terminate()
                self._pool.close()
                self._pool = multiprocessing.Pool(self._n_workers)
                # self._n_pool_resets += 1
                batches = [_load_data(job) for job in batch_jobs]

            ret_vals = []
            for i, b in enumerate(batches):
                if i == 0:
                    for value in b:
                        ret_vals.append([value[None]])
                else:
                    for v_i, value in enumerate(b):
                        ret_vals[v_i].append(value[None])
          
            ret_vals = [np.concatenate(v) for v in ret_vals]
            #todo add time slice here
            if self._hparams.ret_fnames:
                ret_vals = ret_vals + [file_names]

            yield tuple(ret_vals)
    

    def rand_augment(self, batch_data):
          
        scan_vid_fn = lambda a, x: rand_aug(tf.random.uniform(shape=(4,2), minval=0, maxval=100, dtype=tf.int32), x, 
                                            num_layers = 3, magnitude = 10)
        
        aug_vids = tf.scan(scan_vid_fn, batch_data)
        bs = aug_vids.shape[0]
        aug_vids = tf.reshape(aug_vids, (-1,)+tuple(aug_vids.shape[-3:]))
        rc_vids = rand_crop(tf.random.uniform(shape=(4,2), minval=0, maxval=100, dtype=tf.int32), aug_vids, 
                                width=74, height=74, wiggle=10)

        #return tf.reshape(rc_vids, (bs, bl)+tuple(rc_vids.shape[-3:]))
        return tf.reshape(rc_vids, (bs, self.batch_length, 64,64,3))

    def _get_dict(self, *args):
        assert self.batch_length < self._hparams.load_T
        if self._hparams.ret_fnames and self._hparams.load_annotations:
            images, actions, states, annotations, f_names = args
        elif self._hparams.ret_fnames:
            images, actions, states, f_names = args
        elif self._hparams.load_annotations:
            images, actions, states, annotations = args
        else:
            images, actions, states = args
        
        out_dict = {}
        height, width = self._hparams.img_size
        
        if self._hparams.load_random_cam:
            ncam = 1
        else:
            assert len(self._hparams.cams_to_load) == 1
            ncam = len(self._hparams.cams_to_load)
        
        out_dict['image'] = tf.reshape(images, [self.batch_size, self._hparams.load_T, height, width, 3])
       
        #out_dict['images'] = tf.cast(shaped_images, tf.float32) / 255.0
        if self._hparams.color_augmentation:
            out_dict['image'] = color_augment(out_dict['image'], self._hparams.color_augmentation)

        out_dict['action'] = tf.reshape(actions, [self.batch_size, self._hparams.load_T - 1, self._hparams.target_adim])
        #out_dict['states'] = tf.reshape(states, [self.batch_size, self._hparams.load_T, self._hparams.target_sdim])
        #start_idx = np.random.randint(0, self._hparams.load_T - self.batch_length)
        start_idx = tf.random.uniform(shape=(), minval=0, maxval= self._hparams.load_T - self.batch_length, dtype=tf.int32)
        #print(start_idx)
        out_dict['image'] = out_dict['image'][:, start_idx : start_idx + self.batch_length, :, :,:]
        out_dict['image'] = self.rand_augment(out_dict['image'])
        out_dict['action'] = out_dict['action'][:, start_idx : start_idx + self.batch_length, :]
      
        if self._hparams.load_annotations:
            out_dict['annotations'] = tf.reshape(annotations, [self._batch_size, self._hparams.load_T, ncam, height, width, 2])
        if self._hparams.ret_fnames:
            out_dict['f_names'] = f_names
        

        return out_dict