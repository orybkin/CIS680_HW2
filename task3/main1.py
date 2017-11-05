import numpy as np
import tensorflow as tf
from scipy import misc

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
from models import *

def main(config, model):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  train_data_loader, train_label_loader, dummy, dummy = get_loader(
    config.data_path, config.batch_size, config, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader, sample_imgs, sample_labels = get_loader(
      config.data_path, config.batch_size_test, config, 'test', False)
  else:
    test_data_loader, test_label_loader, sample_imgs, sample_labels = get_loader(
      config.data_path, config.batch_size_test,  config, 'test', False)

  if config.train_adversarial:
    n=sample_imgs.shape[0]
    stds=np.zeros(n)
    means=np.zeros(n)
    stds=np.std(sample_imgs.reshape([n,-1]), axis=1)
    means=np.mean(sample_imgs.reshape([n,-1]), axis=1)
    for i in range(n):
      sample_imgs[i]=(sample_imgs[i]-means[i])/stds[i]
      # print(((sample_imgs[i])**2).sum().sum().sum())

  if config.test_adversarial:
    sample_imgs=np.zeros([20,32,32,3])
    sample_labels=np.tile(np.arange(10),[2,1]).T.reshape([-1])
    for i in range(20):
      sample_imgs[i] = misc.imread('selected_imgs/'+str(i)+'_modified.png')

  # print(stds)
  # print(means)

  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader, model, sample_imgs, sample_labels)
  if config.is_train:
    save_config(config)
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

  if config.train_adversarial:
    np.savetxt('perturbed_data/new_labels',(trainer.modif_labels.astype(int)))
    np.savetxt('perturbed_data/new_confidences',(trainer.modif_confidences))
    np.savetxt('perturbed_data/labels',(trainer.sample_labels.astype(int)))
    np.savetxt('perturbed_data/confidences',(trainer.sample_confidences))
    for i in range(n):
      trainer.modif_imgs[i]=(trainer.modif_imgs[i] ) * stds[i] + - means[i]
      misc.imsave('perturbed_imgs/'+str(i)+'_modified.png',trainer.modif_imgs[i])
      misc.imsave('perturbed_imgs/'+str(i)+'_original.png',trainer.sample_imgs[i])
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
  # tf.reset_default_graph()
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # # model=res_net
  # # main(config, model)
  #
  # model=fractal_net
  # main(config, model)

  # Task 3

  # Subtask 1
  # config.load_path = 'task11/model'
  # config.is_train = False
  # config.flip_left_right = True
  # config.normalize_img = True
  # config.pad_and_crop = True
  # config.train_adversarial = True
  # model=last_task
  # main(config, model)


  # Subtask 2
  # config.load_path = 'task11/model'
  # config.is_train = False
  config.flip_left_right = True
  config.normalize_img = True
  config.pad_and_crop = True
  config.train_adversarial = False
  config.test_adversarial = True
  model=only_conv
  main(config, model)