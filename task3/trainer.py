from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange

from models import *

def norm_img(img):
  return img / 127.5 - 1.

def denorm_img(img):
  return (img + 1.) * 127.5

class Trainer(object):
  def __init__(self, config, data_loader, label_loader, test_data_loader, test_label_loader, model, sample_imgs, sample_labels):

    self.normalize_img=config.normalize_img
    self.flip_left_right=config.flip_left_right

    n = sample_imgs.shape[0]
    self.model=model
    self.config=config
    self.sample_imgs=sample_imgs
    self.sample_labels=sample_labels
    self.sample_confidences=np.zeros(n)
    self.eps=1e-7
    self.modif_imgs=sample_imgs.copy()
    self.modif_labels=np.zeros(n)
    self.modif_confidences=np.zeros(n)

    self.config = config
    self.data_loader = data_loader
    self.label_loader = label_loader
    self.test_data_loader = test_data_loader
    self.test_label_loader = test_label_loader
    self.train_adversarial = config.train_adversarial
    self.test_adversarial=config.test_adversarial

    self.optimizer = config.optimizer
    self.batch_size = config.batch_size
    self.batch_size_test = config.batch_size_test

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.test_iter = config.test_iter
    self.wd_ratio = config.wd_ratio

    self.lr = tf.Variable(config.lr, name='lr')

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    self.build_model()
    self.build_test_model()

    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):




    for step in trange(self.start_step, self.max_step):
      fetch_dict = {
        'c_optim': self.c_optim,
        'wd_optim': self.wd_optim,
        'c_loss': self.c_loss,
        'accuracy': self.accuracy }

      if step % self.log_step == self.log_step - 1:
        fetch_dict.update({
          'lr': self.lr,
          'summary': self.summary_op })

      result = self.sess.run(fetch_dict)

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()

        lr = result['lr']
        c_loss = result['c_loss']
        accuracy = result['accuracy']

        print("\n[{}/{}:{:.6f}] Loss_C: {:.6f} Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, c_loss, accuracy))
        sys.stdout.flush()

      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model')

        test_accuracy = 0
        for iter in range(self.test_iter):
          fetch_dict = { "test_accuracy":self.test_accuracy }
          result = self.sess.run(fetch_dict)
          test_accuracy += result['test_accuracy']
        test_accuracy /= self.test_iter

        print("\n[{}/{}:{:.6f}] Test Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, test_accuracy))
        sys.stdout.flush()

        if self.test_adversarial:
          test_accuracy = 0
          for iter in range(self.test_iter):
            fetch_dict = { "test_accuracy":self.test_accuracy }
            result = self.sess.run(fetch_dict,
                                            {self.test_x: self.sample_imgs,
                                             self.test_labels: self.sample_labels})
            test_accuracy += result['test_accuracy']
          test_accuracy /= self.test_iter


          print("\n[{}/{}:{:.6f}] Adversarial Accuracy: {:.4f}" . \
                format(step, self.max_step, lr, test_accuracy))
          sys.stdout.flush()


      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.lr_update])


    # for img in range(20):
    #   for i in range(500):
    #     im_grad, confidence = self.sess.run([self.test_im_grad, self.test_confidence],
    #                                         {self.test_x: self.modif_imgs[img*100:(img+1)*100],
    #                                          self.test_labels: self.sample_labels[img*100:(img+1)*100]})
    #     # im_grad, confidence = self.sess.run([self.test_im_grad, self.test_confidence],
    #     #                                     {self.test_x: np.tile(self.modif_imgs[img], [100, 1, 1, 1]),
    #     #                                      self.test_labels: np.repeat(img, 100)})
    #     im_grad = im_grad[0]
    #     # print(len(self.im_grad), len(im_grad))
    #     # print(np.linalg.norm(im_grad[0]-im_grad))
    #     confidence = confidence[0]
    #     im_grad = im_grad[0]
    #     # if (i % 4999) == 0:
    #       # print(im_grad.dtype, self.modif_imgs.dtype)
    #       # print(np.sign(im_grad)[0:5, 0:5, 0])
    #       # print((im_grad)[0:5,0:5,0])
    #     self.modif_imgs[img] = self.modif_imgs[img] + .0003 * np.sign(im_grad)
    #   self.modif_confidences[img]=confidence.max()
    #   self.modif_labels[img]=confidence.argsort()[-1]
    #   print(self.modif_imgs[img][0:5, 0:5, 0])
    #   print(confidence[0], confidence.max())
    if self.train_adversarial:
      batches=40

      for img in range(batches):
        n=100
        batch=np.arange(img*n,(img+1)*n)
        orig_confidence = self.sess.run(self.test_confidence,
                                            {self.test_x: self.modif_imgs[batch],
                                             self.test_labels: self.sample_labels[batch]})
        # print(orig_confidence[0:5,0:5])
        self.sample_confidences[batch]=orig_confidence[np.arange(0,n),self.sample_labels[batch]]


      for img in range(batches):
        for i in range(100):
          n=100
          batch=np.arange(img*n,(img+1)*n)
          im_grad, confidence = self.sess.run([self.test_im_grad, self.test_confidence],
                                              {self.test_x: self.modif_imgs[batch],
                                               self.test_labels: self.sample_labels[batch]})
          # im_grad, confidence = self.sess.run([self.test_im_grad, self.test_confidence],
          #                                     {self.test_x: np.tile(self.modif_imgs[img], [100, 1, 1, 1]),
          #                                      self.test_labels: np.repeat(img, 100)})
          im_grad = im_grad[0]
          # print(len(self.im_grad), len(im_grad))
          # print(np.linalg.norm(im_grad[0]-im_grad))
          # im_grad = im_grad[0]
          # if (i % 4999) == 0:
            # print(im_grad.dtype, self.modif_imgs.dtype)
            # print(np.sign(im_grad)[0:5, 0:5, 0])
            # print((im_grad)[0:5,0:5,0])
          self.modif_imgs[batch] = self.modif_imgs[batch] + .003 * np.sign(im_grad)
        self.modif_confidences[batch]=confidence.max(axis=1)
        self.modif_labels[batch]=confidence.argmax(axis=1)
        # print(self.modif_imgs[batch][0:5, 0:5, 0])
        # print(np.array([confidence[np.arange(0,n),self.sample_labels[batch]], confidence.max(axis=1)]).T)



            # print(im_grad[0], len(im_grad[0]))

  def build_model(self):
    self.x = self.data_loader
    self.labels = self.label_loader

    if self.normalize_img:
      x=self.x
    else:
      x = norm_img(self.x)

    self.c_loss, feat, self.accuracy, self.c_var, self.im_grad, self.confidence = self.model(
      x, self.labels, self.c_num, self.batch_size,  self.config ,is_train=True, reuse=False)
    c_loss_orig=self.c_loss
    self.c_loss = tf.reduce_mean(self.c_loss, 0)

    # Gather gradients of conv1 & fc4 weights for logging
    with tf.variable_scope("C/conv1", reuse=True):
      conv1_weights = tf.get_variable("weights")
    conv1_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, conv1_weights, self.c_loss)))

    with tf.variable_scope("C/last", reuse=True):
      convlast_weights = tf.get_variable("weights")
    conlvlast_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, convlast_weights, self.c_loss)))

    x_grad = tf.gradients(self.c_loss, x, self.c_loss)
    x_grad = tf.reduce_sum(tf.abs(x_grad[0]), 3, True)
    x_grad = (x_grad - tf.reduce_min(x_grad)) / (tf.reduce_max(x_grad) - tf.reduce_mean(x_grad))
    x_grad = tf.multiply(self.x , x_grad)




    wd_optimizer = tf.train.GradientDescentOptimizer(self.lr)
    if self.optimizer == 'sgd':
      c_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
    elif self.optimizer == 'adam':
      c_optimizer = tf.train.AdamOptimizer(self.lr)
    else:
      raise Exception("[!] Caution! Don't use {} opimizer.".format(self.optimizer))

    for var in tf.trainable_variables():
      weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
      tf.add_to_collection('losses', weight_decay)
    wd_loss = tf.add_n(tf.get_collection('losses'))

    self.c_optim = c_optimizer.minimize(self.c_loss, var_list=self.c_var)
    self.wd_optim = wd_optimizer.minimize(wd_loss)

    self.summary_op = tf.summary.merge([
      tf.summary.scalar("c_loss", self.c_loss),
      tf.summary.scalar("accuracy", self.accuracy),
      tf.summary.scalar("lr", self.lr),
      tf.summary.scalar("conv1_grad", conv1_grad),
      tf.summary.scalar("convlast_grad", conlvlast_grad),

      tf.summary.image("inputs", self.x),
      tf.summary.image("x_grad", x_grad),

      tf.summary.histogram("feature", feat)
    ])

  def test(self):
    self.saver.restore(self.sess, self.model_dir)
    test_accuracy = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_accuracy":self.test_accuracy}
      result = self.sess.run(fetch_dict)
      test_accuracy += result['test_accuracy']
    test_accuracy /= self.test_iter

    print("Accuracy: {:.4f}" . format(test_accuracy))

  def build_test_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader

    if self.normalize_img:
      test_x=self.test_x
    else:
      test_x = norm_img(self.test_x)

    loss, self.test_feat, self.test_accuracy, var, self.test_im_grad, self.test_confidence  = self.model(
      test_x, self.test_labels, self.c_num, self.batch_size_test, self.config ,is_train=False, reuse=True)
