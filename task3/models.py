import tensorflow as tf
from layers import *

def quick_cnn(x, labels, c_num, batch_size, config ,is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32 
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('last', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # fc4
    with tf.variable_scope('fc4', reuse=reuse):
      x = tf.reshape(x, [batch_size, -1])
      x = fc_factory(x, hidden_num, is_train, reuse)
    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc5', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables



def first_task(x, labels,  c_num,batch_size,config , is_train, reuse):
  c_num=10;
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('last', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # fc4
    with tf.variable_scope('fc4', reuse=reuse):
      x = tf.reshape(x, [batch_size, -1])
      x = fc_factory(x, hidden_num, is_train, reuse)
    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc5', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, 1




def only_conv(x, labels, c_num, batch_size, config , is_train, reuse, ):
  c_num=10;
  with tf.variable_scope('C', reuse=reuse) as vs:

    pooling = tf.nn.max_pool
    pool_size=3

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 92
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)

    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
    print(x)

    # conv2
    with tf.variable_scope('conv3', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    print(x)


    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
    print(x)


    with tf.variable_scope('conv5', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
    # fc4

    with tf.variable_scope('conv6', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    feat = x

    print(x)

    with tf.variable_scope('conv7', reuse=reuse):
      #hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
    print(x)

    # fc4

    with tf.variable_scope('conv8', reuse=reuse):
      x = conv_factory(x, hidden_num, 1, 1, is_train, reuse)
    feat = x
    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('last', reuse=reuse):
      kernel_size=1
      stride=1
      W = tf.get_variable('weights', [kernel_size,kernel_size,hidden_num,c_num],
            initializer = tf.contrib.layers.variance_scaling_initializer())
      b = tf.get_variable('biases', [1, 1, 1, c_num],
            initializer = tf.constant_initializer(0.0))

      x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
      x = tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
      x=tf.reshape(x, [batch_size,-1])

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, 1, 1





def only_conv_vanishing(x, labels, c_num, batch_size, config, is_train, reuse, ):
  c_num=10;
  with tf.variable_scope('C', reuse=reuse) as vs:

    pooling = tf.nn.max_pool
    pool_size=3
    batch_norm=False

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 92
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)


    # Uncomment to see vanishing gradients
    for l in range(20):
     with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
       x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)

    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)

    # conv2
    with tf.variable_scope('conv3', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    print(x)

    # conv3
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)


    with tf.variable_scope('conv5', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    # fc4

    with tf.variable_scope('conv6', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    feat = x

    print(x)

    with tf.variable_scope('conv7', reuse=reuse):
      #hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)

    # fc4

    with tf.variable_scope('conv8', reuse=reuse):
      x = conv_factory(x, hidden_num, 1, 1, is_train, reuse,  batch_norm=batch_norm)
    feat = x
    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('last', reuse=reuse):
      kernel_size=1
      stride=1
      W = tf.get_variable('weights', [kernel_size,kernel_size,hidden_num,c_num],
            initializer = tf.contrib.layers.variance_scaling_initializer())
      b = tf.get_variable('biases', [1, 1, 1, c_num],
            initializer = tf.constant_initializer(0.0))

      x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
      x = tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
      x=tf.reshape(x, [batch_size,-1])

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, 1



def res_net(x, labels, c_num, batch_size, config, is_train, reuse, ):
  c_num=10;
  with tf.variable_scope('C', reuse=reuse) as vs:

    pooling = tf.nn.max_pool
    pool_size=3
    batch_norm=False

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 64
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)

    # x_first=x

    # Uncomment to see vanishing gradients
    for l in range(10):
      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
        x_begin = x
        with tf.variable_scope('1', reuse=reuse):
          x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
        with tf.variable_scope('2', reuse=reuse):
          x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
        x = x + x_begin

    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm) # + x_first # skip-connection

    print(x)

    # conv2
    with tf.variable_scope('conv3', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    print(x)

    # conv3
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)


    with tf.variable_scope('conv5', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    # fc4

    with tf.variable_scope('conv6', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    feat = x

    print(x)

    with tf.variable_scope('conv7', reuse=reuse):
      #hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)

    # fc4

    with tf.variable_scope('conv8', reuse=reuse):
      x = conv_factory(x, hidden_num, 1, 1, is_train, reuse,  batch_norm=batch_norm)
    feat = x
    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('last', reuse=reuse):
      kernel_size=1
      stride=1
      W = tf.get_variable('weights', [kernel_size,kernel_size,hidden_num,c_num],
            initializer = tf.contrib.layers.variance_scaling_initializer())
      b = tf.get_variable('biases', [1, 1, 1, c_num],
            initializer = tf.constant_initializer(0.0))

      x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
      x = tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
      x=tf.reshape(x, [batch_size,-1])

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, 1, 1





def fractal_net(x, labels, c_num, batch_size, config, is_train, reuse, ):
  c_num=10;
  with tf.variable_scope('C', reuse=reuse) as vs:

    pooling = tf.nn.max_pool
    pool_size=3
    batch_norm=False

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 64
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)

    # Uncomment to see vanishing gradients
    for l in range(5):
      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
        x_begin = x

        with tf.variable_scope('path1', reuse=reuse):
          x1 = conv_factory(x_begin, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)


        with tf.variable_scope('path11', reuse=reuse):
          with tf.variable_scope('1', reuse=reuse):
            x11 = conv_factory(x_begin, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
          with tf.variable_scope('11', reuse=reuse):
            x111 = conv_factory(x_begin, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
          with tf.variable_scope('12', reuse=reuse):
            x112 = conv_factory(x111, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)

          x_middle=(x11+x112)/2


        with tf.variable_scope('path12', reuse=reuse):
          with tf.variable_scope('1', reuse=reuse):
            x11 = conv_factory(x_middle, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
          with tf.variable_scope('11', reuse=reuse):
            x111 = conv_factory(x_middle, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
          with tf.variable_scope('12', reuse=reuse):
            x112 = conv_factory(x111, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)

        x=(x1+x11+x112)/3

    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)  # skip-connection

    print(x)

    # conv2
    with tf.variable_scope('conv3', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    print(x)

    # conv3
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)


    with tf.variable_scope('conv5', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    # fc4

    with tf.variable_scope('conv6', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
      x = pooling(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='SAME')
    feat = x

    print(x)

    with tf.variable_scope('conv7', reuse=reuse):
      #hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse,  batch_norm=batch_norm)
    print(x)

    # fc4

    with tf.variable_scope('conv8', reuse=reuse):
      x = conv_factory(x, hidden_num, 1, 1, is_train, reuse,  batch_norm=batch_norm)
    feat = x
    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('last', reuse=reuse):
      kernel_size=1
      stride=1
      W = tf.get_variable('weights', [kernel_size,kernel_size,hidden_num,c_num],
            initializer = tf.contrib.layers.variance_scaling_initializer())
      b = tf.get_variable('biases', [1, 1, 1, c_num],
            initializer = tf.constant_initializer(0.0))

      x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
      x = tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
      x=tf.reshape(x, [batch_size,-1])

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, 1, 1




def last_task(x, labels,  c_num,batch_size,config , is_train, reuse):
  c_num=10;
  batch_norm=False
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1

    x_copy=x

    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('last', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse,  batch_norm=batch_norm)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    print(x)
    # fc4
    with tf.variable_scope('fc4', reuse=reuse):
      x = tf.reshape(x, [-1, 4*4*64])
      x = fc_factory(x, hidden_num, is_train, reuse,  batch_norm=batch_norm)

    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc5', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      print('loss', loss)
      confidence=tf.nn.softmax(x)
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  im_grad = tf.gradients(loss, x_copy, loss)
  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, im_grad, confidence


# operate on denormalized
# use original network
# select best
# select only high original confidence