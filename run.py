from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
import tensorflow as tf
from matplotlib import pyplot as plt

# load and filter robonet database
database = load_metadata('~/research/RoboNet/hdf5')
database = database[database['robot'] == 'franka']

# create data loader object and grab train/val image tensors
robonetdataset = RoboNetDataset(batch_size=16, dataset_files_or_metadata=database)

train_dataset = robonetdataset.mode_generators['train']
val_dataset = robonetdataset.mode_generators['val']
test_dataset = robonetdataset.mode_generators['test']


num_train_files = robonetdataset.mode_num_files['train']
num_val_files = robonetdataset.mode_num_files['val']
num_test_files = robonetdataset.mode_num_files['test']

for i, elem in enumerate(val_dataset):
    #if i >= 5:
    #    break
    print(elem['images'].shape, elem['actions'].shape)
    #plt.imsave('cam1/out'+str(i)+'.png', elem['images'][0,0,0].numpy() )
  