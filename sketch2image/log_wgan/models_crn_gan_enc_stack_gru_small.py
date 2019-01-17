import functools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.python.framework import ops

from resnet_rnn import resnet_block, resnet_deconv_block, resnet_conv, resnet_deconv, upsample_conv, mean_pool, unrolled_lstm_conv, unrolled_lstm_deconv, unrolled_gru_conv, unrolled_gru_deconv


print('small')
USE_BOTTLENECK = False
SIZE = 64
NUM_BLOCKS = 1
CRAMER = False


def one_hot_to_dense(labels):
    # Assume on value is 1
    batch_size = int(labels.get_shape()[0])
    return tf.reshape(tf.where(tf.equal(labels, 1))[:, 1], (batch_size,))


def batchnorm(inputs, data_format=None, activation_fn=None, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if data_format != 'NCHW':
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, (0, 2, 3), keep_dims=True)
    shape = mean.get_shape().as_list()  # shape is [1,n,1,1]
    offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, shape[1]], dtype='float32'))
    scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:, :, None, None], scale[:, :, None, None], 1e-5)
    return result


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)


def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        leak = tf.get_variable("param", shape=None, initializer=0.2, regularizer=None,
                               trainable=True, caching_device=None)
        return tf.maximum(leak * x, x)


def miu_relu(x, miu=0.7, name="miu_relu"):
    with tf.variable_scope(name):
        return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def p_miu_relu(x, name="p_miu_relu"):
    with tf.variable_scope(name):
        miu = tf.get_variable("param_miu", shape=None, initializer=0.7, regularizer=None,
                              trainable=True, caching_device=None)
        return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def matsushita_entropy(x, name="matsushita_entropy"):
    with tf.variable_scope(name):
        return (1 + x / tf.sqrt(1 + x ** 2)) / 2.


def image_encoder_s1_gru(x, num_classes, reuse=False, data_format='NCHW', labels=None, scope_name=None):
    print("CONV_GRU")
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear

    if normalizer_params_e is not None and normalizer_fn_e != ly.batch_norm and normalizer_fn_e != ly.layer_norm:
        normalizer_params_e['labels'] = labels
        normalizer_params_e['n_labels'] = num_classes

    if data_format == 'NCHW':
        resized_x = []
        resized_ = x
        resized_x.append(resized_)

        for i in range(4):
            resized_ = mean_pool(resized_, data_format=data_format)
            resized_x.append(resized_)
        resized_x = resized_x[::-1]
    else:
        raise NotImplementedError

    output_list = []

    # with tf.variable_scope(scope_name) as scope:
    #     if reuse:
    #         scope.reuse_variables()
    x_list = resized_x

    h0 = ly.conv2d(x_list[-1], size * 1, kernel_size=7, stride=2, data_format=data_format,
                   activation_fn=activation_fn_e,
                   normalizer_fn=normalizer_fn_e,
                   normalizer_params=normalizer_params_e,
                   weights_initializer=weight_initializer)

    # Initial memory state
    hidden_state_shape = h0.get_shape().as_list()
    batch_size = hidden_state_shape[0]
    hidden_state_shape[0] = 1
    hts_0 = [h0]
    for i in range(1, num_blocks):
        h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                     initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
        hts_0.append(h0)

    hts_1 = unrolled_gru_conv(x_list[-2], hts_0,
                              size * 1, stride=2, dilate_rate=1,
                              data_format=data_format, num_blocks=num_blocks,
                              first_unit=True, last_unit=False,
                              activation_fn=activation_fn_e,
                              normalizer_fn=normalizer_fn_e,
                              normalizer_params=normalizer_params_e,
                              weights_initializer=weight_initializer,
                              use_bottleneck=USE_BOTTLENECK,
                              unit_num=1)
    output_list.append(hts_1[-1])
    hts_2 = unrolled_gru_conv(x_list[-3], hts_1,
                              size * 2, stride=2, dilate_rate=1,
                              data_format=data_format, num_blocks=num_blocks,
                              first_unit=False, last_unit=False,
                              activation_fn=activation_fn_e,
                              normalizer_fn=normalizer_fn_e,
                              normalizer_params=normalizer_params_e,
                              weights_initializer=weight_initializer,
                              use_bottleneck=USE_BOTTLENECK,
                              unit_num=2)
    output_list.append(hts_2[-1])
    hts_3 = unrolled_gru_conv(x_list[-4], hts_2,
                              size * 4, stride=2, dilate_rate=1,
                              data_format=data_format, num_blocks=num_blocks,
                              first_unit=False, last_unit=False,
                              activation_fn=activation_fn_e,
                              normalizer_fn=normalizer_fn_e,
                              normalizer_params=normalizer_params_e,
                              weights_initializer=weight_initializer,
                              use_bottleneck=USE_BOTTLENECK,
                              unit_num=3)
    output_list.append(hts_3[-1])
    hts_4 = unrolled_gru_conv(x_list[-5], hts_3,
                              size * 8, stride=2, dilate_rate=1,
                              data_format=data_format, num_blocks=num_blocks,
                              first_unit=False, last_unit=True,
                              activation_fn=activation_fn_e,
                              normalizer_fn=normalizer_fn_e,
                              normalizer_params=normalizer_params_e,
                              weights_initializer=weight_initializer,
                              use_bottleneck=USE_BOTTLENECK,
                              unit_num=4)
    output_list.append(hts_4[-1])

    return output_list


# GRU
def generator_l_s1_skip(z, output_channel, num_classes, reuse=False, data_format='NCHW',
                        labels=None, scope_name=None):
    print("DECONV_GRU")
    size = SIZE
    num_blocks = NUM_BLOCKS

    input_dims = z.get_shape().as_list()
    resize_func = tf.image.resize_area

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
        z_orig = tf.identity(z)
        z = tf.transpose(z, [0, 2, 3, 1])
        resized_z = [
            tf.transpose(resize_func(z, [int(height / 32), int(width / 32)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(z, [int(height / 16), int(width / 16)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(z, [int(height / 8), int(width / 8)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(z, [int(height / 4), int(width / 4)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(z, [int(height / 2), int(width / 2)]), [0, 3, 1, 2]),
        ]
        z = z_orig
    else:
        height = input_dims[1]
        width = input_dims[2]
        resized_z = [
            resize_func(z, [int(height / 32), int(width / 32)]),
            resize_func(z, [int(height / 16), int(width / 16)]),
            resize_func(z, [int(height / 8), int(width / 8)]),
            resize_func(z, [int(height / 4), int(width / 4)]),
            resize_func(z, [int(height / 2), int(width / 2)]),
        ]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    output_list = []

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        z_encoded = image_encoder_s1_gru(z, num_classes=num_classes, reuse=reuse, data_format=data_format,
                                         labels=labels, scope_name=scope_name)

        input_e_dims = z_encoded[-1].get_shape().as_list()
        input_e_dims[concat_axis] = int(input_e_dims[concat_axis] / 2.)

        noise = tf.random_normal(shape=(input_e_dims[0], 256), dtype=tf.float32)
        noise = ly.fully_connected(noise, int(np.prod(input_e_dims[1:])), activation_fn=activation_fn_g)
        noise = tf.reshape(noise, shape=input_e_dims)

        # Initial memory state
        hidden_state_shape = z_encoded[-1].get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [z_encoded[-1]]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.random_normal_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        input_0 = tf.concat([resized_z[0], noise], axis=concat_axis)
        hts_1 = unrolled_gru_deconv(input_0, hts_0,
                                    size * 6, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=True, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=0)
        # output_list.append(ly.conv2d(hts_1[-1], 3, 3, stride=1, data_format=data_format,
        #                              normalizer_fn=None, activation_fn=tf.nn.tanh,
        #                              weights_initializer=weight_initializer))
        input_1 = tf.concat([resized_z[1], z_encoded[-2]], axis=concat_axis)
        hts_2 = unrolled_gru_deconv(input_1, hts_1,
                                    size * 4, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=2)
        # output_list.append(ly.conv2d(hts_2[-1], 3, 3, stride=1, data_format=data_format,
        #                              normalizer_fn=None, activation_fn=tf.nn.tanh,
        #                              weights_initializer=weight_initializer))
        input_2 = tf.concat([resized_z[2], z_encoded[-3]], axis=concat_axis)
        hts_3 = unrolled_gru_deconv(input_2, hts_2,
                                    size * 2, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=4)
        # output_list.append(ly.conv2d(hts_3[-1], 3, 3, stride=1, data_format=data_format,
        #                              normalizer_fn=None, activation_fn=tf.nn.tanh,
        #                              weights_initializer=weight_initializer))
        input_3 = tf.concat([resized_z[3], z_encoded[-4]], axis=concat_axis)
        hts_4 = unrolled_gru_deconv(input_3, hts_3,
                                    size * 2, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=6)
        # output_list.append(ly.conv2d(hts_4[-1], 3, 3, stride=1, data_format=data_format,
        #                              normalizer_fn=None, activation_fn=tf.nn.tanh,
        #                              weights_initializer=weight_initializer))
        hts_5 = unrolled_gru_deconv(resized_z[4], hts_4,
                                    size * 1, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=True,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=8)
        output_list.append(ly.conv2d(hts_5[-1], 3, 7, stride=1, data_format=data_format,
                                     normalizer_fn=None, activation_fn=tf.nn.tanh,
                                     weights_initializer=weight_initializer))
        # out = ly.conv2d(train, output_channel, 7, stride=1, data_format=data_format,
        #                 activation_fn=tf.nn.tanh, weights_initializer=weight_initializer)
        assert output_list[-1].get_shape().as_list()[2] == 64
        return output_list


# GRU
def generator_l_s2(z, extra, output_channel, num_classes, reuse=False, data_format='NCHW',
                   labels=None, scope_name=None):
    print("DECONV_GRU")
    size = SIZE
    num_blocks = NUM_BLOCKS

    if type(z) is list:
        z = z[-1]

    input_dims = extra.get_shape().as_list()
    resize_func = tf.image.resize_area

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
        extra_orig = tf.identity(extra)
        extra = tf.transpose(extra, [0, 2, 3, 1])
        resized_extra = [
            tf.transpose(resize_func(extra, [int(height / 32), int(width / 32)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(extra, [int(height / 16), int(width / 16)]), [0, 3, 1, 2]),
            # tf.transpose(resize_func(extra, [int(height / 8), int(width / 8)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(extra, [int(height / 8), int(width / 8)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(extra, [int(height / 4), int(width / 4)]), [0, 3, 1, 2]),
            tf.transpose(resize_func(extra, [int(height / 2), int(width / 2)]), [0, 3, 1, 2]),
        ]
        extra = extra_orig
    else:
        raise NotImplementedError
        height = input_dims[1]
        width = input_dims[2]
        resized_extra = [
            # resize_func(extra, [int(height / 32), int(width / 32)]),
            # resize_func(extra, [int(height / 16), int(width / 16)]),
            resize_func(extra, [int(height / 8), int(width / 8)]),
            resize_func(extra, [int(height / 4), int(width / 4)]),
            resize_func(extra, [int(height / 2), int(width / 2)]),
        ]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    output_list = []

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        z_encoded = image_encoder_s2(z, num_classes=num_classes, reuse=reuse, data_format=data_format,
                                     labels=labels, scope_name=scope_name)

        # Initial memory state
        hidden_state_shape = z_encoded.get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [z_encoded]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.random_normal_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        hts_1 = unrolled_gru_deconv(resized_extra[0], hts_0,
                                    size * 8, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=True, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=0)
        # hts_1 = unrolled_gru_deconv(resized_extra[0], hts_1,
        #                             size * 8, stride=1, data_format=data_format, num_blocks=num_blocks,
        #                             first_unit=False, last_unit=False,
        #                             activation_fn=activation_fn_g,
        #                             normalizer_fn=normalizer_fn_g,
        #                             normalizer_params=normalizer_params_g,
        #                             weights_initializer=weight_initializer,
        #                             use_bottleneck=USE_BOTTLENECK,
        #                             unit_num=1)
        hts_1 = unrolled_gru_deconv(resized_extra[1], hts_1,
                                    size * 8, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=2)
        hts_2 = unrolled_gru_deconv(resized_extra[2], hts_1,
                                    size * 4, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=11)
        hts_3 = unrolled_gru_deconv(resized_extra[3], hts_2,
                                    size * 2, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=False,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=12)
        hts_4 = unrolled_gru_deconv(resized_extra[4], hts_3,
                                    size * 1, stride=2, data_format=data_format, num_blocks=num_blocks,
                                    first_unit=False, last_unit=True,
                                    activation_fn=activation_fn_g,
                                    normalizer_fn=normalizer_fn_g,
                                    normalizer_params=normalizer_params_g,
                                    weights_initializer=weight_initializer,
                                    use_bottleneck=USE_BOTTLENECK,
                                    unit_num=13)
        output_list.append(ly.conv2d(hts_4[-1], 3, 7, stride=1, data_format=data_format,
                                     normalizer_fn=None, activation_fn=tf.nn.tanh,
                                     weights_initializer=weight_initializer))
        print("G_s2 out: %d" % output_list[-1].get_shape().as_list()[2])
        return output_list


# GRU
def critic_l_multiple_s1(x, num_classes, reuse=False, data_format='NCHW', scope_name=None, cramer=CRAMER):
    print("CONV_GRU")
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3
    if type(x) is list:
        x = x[-1]

    # if cond is not None:
    #     x = tf.concat([x, cond], axis=concat_axis)

    if data_format == 'NCHW':
        resized_x = []
        resized_ = x
        resized_x.append(resized_)

        for i in range(4):
            resized_ = mean_pool(resized_, data_format=data_format)
            resized_x.append(resized_)
        resized_x = resized_x[::-1]
    else:
        raise NotImplementedError

    output_list = []
    output_dim = 256 if cramer else 1

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        x_list = resized_x

        h0 = ly.conv2d(x_list[-1], 6, kernel_size=7, stride=1, data_format=data_format,
                       activation_fn=activation_fn_d,
                       normalizer_fn=normalizer_fn_d,
                       normalizer_params=normalizer_params_d,
                       weights_initializer=weight_initializer)

        # Initial memory state
        hidden_state_shape = h0.get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [h0]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        hts_1 = unrolled_gru_conv(x_list[-1], hts_0,
                                  size * 2, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=True, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=1)
        hts_2 = unrolled_gru_conv(x_list[-2], hts_1,
                                  size * 4, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=2)
        hts_3 = unrolled_gru_conv(x_list[-3], hts_2,
                                  size * 8, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=3)
        hts_4 = unrolled_gru_conv(x_list[-4], hts_3,
                                  size * 16, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=True,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=4)

        img = hts_4[-1]

        # discriminator end
        disc = ly.conv2d(img, output_dim, kernel_size=1, stride=1, data_format=data_format,
                         activation_fn=None, normalizer_fn=None,
                         weights_initializer=weight_initializer)

        # classification end
        img = tf.reduce_mean(img, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
        logits = ly.fully_connected(img, num_classes, activation_fn=None, normalizer_fn=None)

    return disc, logits


# GRU
def critic_l_multiple_s2(x, num_classes, reuse=False, data_format='NCHW', scope_name=None, cramer=CRAMER):
    print("CONV_GRU")
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3
    if type(x) is list:
        x = x[-1]

    # if cond is not None:
    #     x = tf.concat([x, cond], axis=concat_axis)

    if data_format == 'NCHW':
        resized_x = []
        resized_ = x
        resized_x.append(resized_)
        resized_ = mean_pool(resized_, data_format=data_format)
        for i in range(6):
            resized_ = mean_pool(resized_, data_format=data_format)
            resized_x.append(resized_)
        resized_x = resized_x[::-1]
    else:
        raise NotImplementedError

    output_list = []
    output_dim = 256 if cramer else 1

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        x_list = resized_x

        h0 = ly.conv2d(x_list[-1], 6, kernel_size=7, stride=2, data_format=data_format,
                       activation_fn=activation_fn_d,
                       normalizer_fn=normalizer_fn_d,
                       normalizer_params=normalizer_params_d,
                       weights_initializer=weight_initializer)

        # Initial memory state
        hidden_state_shape = h0.get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [h0]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        inp_0 = ly.conv2d(x_list[-1], 6, kernel_size=7, stride=2, data_format=data_format,
                          activation_fn=activation_fn_d,
                          normalizer_fn=normalizer_fn_d,
                          normalizer_params=normalizer_params_d,
                          weights_initializer=weight_initializer)
        hts_1 = unrolled_gru_conv(inp_0, hts_0,
                                  size * 1, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=True, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=1)
        hts_2 = unrolled_gru_conv(x_list[-2], hts_1,
                                  size * 2, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=2)
        hts_3 = unrolled_gru_conv(x_list[-3], hts_2,
                                  size * 4, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=3)
        hts_4 = unrolled_gru_conv(x_list[-4], hts_3,
                                  size * 8, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=4)
        hts_5 = unrolled_gru_conv(x_list[-5], hts_4,
                                  size * 16, stride=2, dilate_rate=1,
                                  data_format=data_format, num_blocks=num_blocks,
                                  first_unit=False, last_unit=False,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer,
                                  use_bottleneck=USE_BOTTLENECK,
                                  unit_num=5)
        # hts_6 = unrolled_gru_conv(x_list[-6], hts_5,
        #                           size * 16, stride=2, dilate_rate=1,
        #                           data_format=data_format, num_blocks=num_blocks,
        #                           first_unit=False, last_unit=True,
        #                           activation_fn=activation_fn_d,
        #                           normalizer_fn=normalizer_fn_d,
        #                           normalizer_params=normalizer_params_d,
        #                           weights_initializer=weight_initializer,
        #                           use_bottleneck=USE_BOTTLENECK,
        #                           unit_num=6)

        img = hts_5[-1]

        # img = tf.concat(output_list, axis=concat_axis)

        # img = tf.add_n(
        #     [img[:, :, ::2, ::2], img[:, :, 1::2, ::2], img[:, :, ::2, 1::2], img[:, :, 1::2, 1::2]]) / 4.
        # discriminator end
        disc = ly.conv2d(img, output_dim, kernel_size=1, stride=1, data_format=data_format,
                         activation_fn=None, normalizer_fn=None,
                         weights_initializer=weight_initializer)

        # classification end
        img = tf.reduce_mean(img, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
        logits = ly.fully_connected(img, num_classes, activation_fn=None, normalizer_fn=None)

    return disc, logits


weight_initializer = tf.random_normal_initializer(0, 0.02)
# weight_initializer = ly.xavier_initializer_conv2d()


def set_param(data_format='NCHW'):
    global model_data_format, normalizer_fn_e, normalizer_fn_g, normalizer_fn_d, normalizer_fn_ce,\
        normalizer_params_e, normalizer_params_g, normalizer_params_d, normalizer_params_ce
    model_data_format = data_format
    # normalizer_fn_e = ly.batch_norm
    # normalizer_params_e = {'fused': True, 'data_format': model_data_format,
    #                        'is_training': True}
    normalizer_fn_e = batchnorm
    normalizer_params_e = {'data_format': model_data_format}
    normalizer_fn_g = batchnorm
    normalizer_params_g = {'data_format': model_data_format}
    # normalizer_fn_e = None
    # normalizer_params_e = None
    # normalizer_fn_g = None
    # normalizer_params_g = None
    # normalizer_fn_g = ly.layer_norm
    # normalizer_params_g = None
    normalizer_fn_d = None
    normalizer_params_d = None
    normalizer_fn_ce = None
    normalizer_params_ce = None


model_data_format = None


normalizer_fn_e = ly.batch_norm
normalizer_params_e = {'fused': True, 'data_format': model_data_format,
                       'is_training': True}
# normalizer_params_e = {'fused': True, 'data_format': model_data_format,
#                        'is_training': True, 'decay': 0.95}
normalizer_fn_g = ly.batch_norm
normalizer_params_g = {'fused': True, 'data_format': model_data_format,
                       'is_training': True}
# normalizer_params_g = {'fused': True, 'data_format': model_data_format,
#                        'is_training': True, 'decay': 0.95}
normalizer_fn_d = None
normalizer_params_d = None

normalizer_fn_ce = None
normalizer_params_ce = None

activation_fn_e = miu_relu
activation_fn_g = miu_relu
activation_fn_d = prelu
print('prelu')
activation_fn_d_last = None
# activation_fn_d_last = None
# activation_fn_ce = prelu

generator_s1 = generator_l_s1_skip
generator_s2 = generator_l_s2
critic_s1 = critic_l_multiple_s1
critic_s2 = critic_l_multiple_s2
# critic_e = critic_e_fc
