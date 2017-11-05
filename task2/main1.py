import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
from models import *

def main(config, model):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  train_data_loader, train_label_loader = get_loader(
    config.data_path, config.batch_size, config, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, config, 'test', False)
  else:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test,  config, config.split, False)

  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader, model)
  if config.is_train:
    save_config(config)
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

  # tf.Session.reset(trainer.sess)

if __name__ == "__main__":
  config, unparsed = get_config()
  # README: IMPLEMENTATION NOTE
  # TO TRAIN THE ACTUAL NETWORKS YOU NEED TO UNCOMMENT THE RELEVANT PART OF THE CODE BELOW
  # YOU NEED TO UNCOMMENT THEM ONE AT A TIME BECAUSE I HAVEN'T FIGURED OUT HOW TO RESET THE TENSORFLOW CONTEXT
  # Task 1

  # # Subtask 1
  # tf.reset_default_graph()
  # config.flip_left_right=False
  # config.normalize_img = False
  # config.pad_and_crop=False
  # model=first_task
  # main(config, model)
  #
  #
  # # Subtask 2
  # tf.reset_default_graph()
  # config.flip_left_right=False
  # config.normalize_img=True
  # config.pad_and_crop=False
  # model=first_task
  # main(config, model)

  # # Subtask 3
  # tf.reset_default_graph()
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = False
  # model=first_task
  # main(config, model)

  # Subtask 4
  # tf.reset_default_graph()
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # model=first_task
  # main(config, model)
  #
  #
  # # Task 2
  #
  # Subtask 1
  # tf.reset_default_graph()
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # model=only_conv
  # main(config, model)
  #
  #
  # # Subtask 3
  # tf.reset_default_graph()
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # model=only_conv_vanishing
  # main(config, model)


  # Subtask 4
  tf.reset_default_graph()
  config.flip_left_right = True
  config.normalize_img = True
  config.pad_and_crop = True
  # model=res_net
  # main(config, model)

  model=fractal_net
  main(config, model)

  # Task 3

  # Subtask 1
  # config.load_path = 'task11/model'
  # config.is_train = False
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # model=last_task
  # main(config, model)