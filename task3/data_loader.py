import os
import numpy as np
import tensorflow as tf
from scipy import misc




def read_labeled_image_list(img_list_path, img_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_list_path: a .txt file with one /path/to/image with one label per line
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  f = open(img_list_path, 'r')
  img_paths = []
  img_names= []
  labs = []
  for line in f:
    img_name, lab = line[:-1].split(' ')
    img_paths.append(img_dir + img_name)
    img_names.append(img_name)
    labs.append(int(lab))

  # print(np.unique(labs))
  idx=np.argsort(labs).reshape([10,1000])
  # print(np.array(labs)[idx])

  # sample_idx=idx[:,0:2].reshape([-1])
  sample_idx=idx[:,0:400].reshape([-1])

  # print(np.array(labs)[sample_idx.reshape([-1])])
  # print([sample_idx.reshape([-1])])

  f.close()
  return img_paths, labs, sample_idx #, np.array(img_names)[sample_idx]

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  lab = input_queue[1]
  img_path = tf.read_file(input_queue[0])
  img = tf.image.decode_png(img_path, channels=3)
  return img, lab

def get_loader(root, batch_size, config, split=None, shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    split: Data for train/val/test, a string
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """
  img_paths_np, labs_np, sample_idx = read_labeled_image_list(root+'/devkit/'+split+'.txt', root+'/imgs/')
  img_paths_np_numpy=np.array(img_paths_np)
  # print(img_paths_np_numpy)
  n=sample_idx.shape[0]


  sample_imgs=np.zeros([n,32,32,3])
  sample_labels=np.array(labs_np)[sample_idx]
  for i in range(n):
    sample_imgs[i] = misc.imread(img_paths_np_numpy[sample_idx][i])

  # print(sample_img)

  with tf.device('/cpu:0'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
    labs = tf.convert_to_tensor(labs_np, dtype=tf.int64)

    input_queue = tf.train.slice_input_producer([img_paths, labs],
                  shuffle=shuffle, capacity=10*batch_size)

    img, lab = read_images_from_disk(input_queue)

    img.set_shape([32, 32, 3])
    img = tf.cast(img, tf.float32)

    if config.normalize_img:
        img = tf.image.per_image_standardization(img)

    if (split=='train'):
      if config.flip_left_right:
        img=tf.image.random_flip_left_right(img)
      if config.pad_and_crop:
        img = tf.pad(img, [[4,4],[4,4],[0,0]])
        img = tf.random_crop(img, [32,32,3])

    img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1,
                           batch_size=batch_size, capacity=10*batch_size)

  return img_batch, lab_batch, sample_imgs.astype(np.float32), sample_labels
