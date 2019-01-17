import tensorflow as tf
import tensorflow.contrib.layers as ly

USE_ALL_DECONV = False


def mean_pool(input, data_format):
    assert data_format == 'NCHW'
    output = tf.add_n(
        [input[:, :, ::2, ::2], input[:, :, 1::2, ::2], input[:, :, ::2, 1::2], input[:, :, 1::2, 1::2]]) / 4.
    return output


def upsample(input, data_format):
    assert data_format == 'NCHW'
    output = tf.concat([input, input, input, input], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def conv_mean_pool(inputs, num_outputs, kernel_size, rate=1,
                   activation_fn=None,
                   normalizer_fn=None, normalizer_params=None,
                   weights_regularizer=None,
                   weights_initializer=ly.xavier_initializer_conv2d(),
                   biases_initializer=tf.zeros_initializer(),
                   data_format='NCHW'):
    output = ly.conv2d(inputs, num_outputs, kernel_size, rate=rate, activation_fn=activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                       weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                       biases_initializer=biases_initializer,
                       data_format=data_format)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def mean_pool_conv(inputs, num_outputs, kernel_size, rate=1,
                   activation_fn=None,
                   normalizer_fn=None, normalizer_params=None,
                   weights_regularizer=None,
                   weights_initializer=ly.xavier_initializer_conv2d(),
                   data_format='NCHW'):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = ly.conv2d(output, num_outputs, kernel_size, rate=rate, activation_fn=activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                       weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                       data_format=data_format)
    return output


def upsample_conv(inputs, num_outputs, kernel_size, activation_fn=None,
                  normalizer_fn=None, normalizer_params=None,
                  weights_regularizer=None,
                  weights_initializer=ly.xavier_initializer_conv2d(),
                  biases_initializer=tf.zeros_initializer(),
                  data_format='NCHW'):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1 if data_format == 'NCHW' else 3)
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 3, 1, 2])
    output = ly.conv2d(output, num_outputs, kernel_size, activation_fn=activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                       weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                       biases_initializer=biases_initializer,
                       data_format=data_format)
    return output


def upsample_conv_bilinear(inputs, num_outputs, kernel_size, activation_fn=None,
                           normalizer_fn=None, normalizer_params=None,
                           weights_regularizer=None,
                           weights_initializer=ly.xavier_initializer_conv2d(),
                           data_format='NCHW'):
    output = inputs
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 2, 3, 1])
    batch_size, height, width, channel = [int(i) for i in output.get_shape()]
    # output = tf.Print(output, [tf.reduce_min(output), tf.reduce_max(output)], message='before')
    output = tf.image.resize_bilinear(output, [height * 2, width * 2])
    # output = tf.Print(output, [tf.reduce_min(output), tf.reduce_max(output)], message='after')
    slice0 = output[:, :, :, 0::4]
    slice1 = output[:, :, :, 1::4]
    slice2 = output[:, :, :, 2::4]
    slice3 = output[:, :, :, 3::4]
    output = slice0 + slice1 + slice2 + slice3
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 3, 1, 2])
    output = ly.conv2d(output, num_outputs, kernel_size, activation_fn=activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                       weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                       data_format=data_format)
    return output


def resnet_block(x, filter_depth, stride,
                 dilate=1, first_layer_no_dilate=False,
                 norm_activate_first=True,
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=ly.xavier_initializer_conv2d(),
                 data_format='NCHW',
                 weight_decay_rate=1e-8):
    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)
    in_filter_depth = int(x.get_shape()[channel_index])
    # assert in_filter_depth <= filter_depth

    orig_x = tf.identity(x)
    if norm_activate_first:
        with tf.variable_scope('first_normalization'):
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                x = normalizer_fn(x, activation_fn=None, **normalizer_params)
            x = activation_fn(x)

    with tf.variable_scope('subs'):
        # if extra is not None:
        #     x = tf.concat([x, extra], axis=channel_index)
        x = ly.conv2d(x, filter_depth, 3, stride=1,
                      rate=1 if first_layer_no_dilate else dilate,
                      data_format=data_format,
                      normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                      activation_fn=activation_fn, weights_regularizer=regularizer,
                      weights_initializer=weights_initializer)
        if stride == 2:
            x = conv_mean_pool(x, filter_depth, 3,
                               rate=dilate,
                               activation_fn=None, normalizer_fn=None,
                               data_format=data_format, weights_regularizer=regularizer,
                               weights_initializer=weights_initializer)
        else:
            x = ly.conv2d(x, filter_depth, 3, stride=stride,
                          rate=dilate,
                          data_format=data_format, normalizer_fn=None,
                          activation_fn=None, weights_regularizer=regularizer,
                          weights_initializer=weights_initializer)

    with tf.variable_scope('sub_add'):
        if stride == 2:
            orig_x = conv_mean_pool(orig_x, filter_depth, 1,
                                    rate=dilate,
                                    data_format=data_format,
                                    activation_fn=None,
                                    weights_regularizer=regularizer,
                                    weights_initializer=weights_initializer)
        else:
            orig_x = ly.conv2d(orig_x, filter_depth, 1, stride=stride,
                               rate=dilate,
                               data_format=data_format,
                               activation_fn=None,
                               weights_regularizer=regularizer,
                               weights_initializer=weights_initializer)
        x += orig_x

    return x


def lstm_conv_block(inp, ht, ct, filter_depth, stride,
                    dilate=1, first_layer_no_dilate=None,
                    norm_activate_first=None,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=ly.xavier_initializer_conv2d(),
                    data_format='NCHW',
                    weight_decay_rate=1e-8, norm_mask=False):
    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    full_inp = tf.concat([inp, ht], axis=channel_index)
    if stride == 2:
        # input gate
        ig = conv_mean_pool(full_inp, filter_depth, 3, rate=1, data_format=data_format,
                            normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                            activation_fn=tf.nn.sigmoid,
                            weights_regularizer=regularizer,
                            weights_initializer=weights_initializer)
        # forget gate
        fg = conv_mean_pool(full_inp, filter_depth, 3, rate=1, data_format=data_format,
                            normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                            activation_fn=tf.nn.sigmoid,
                            weights_regularizer=regularizer,
                            weights_initializer=weights_initializer)
        # output gate
        og = conv_mean_pool(full_inp, filter_depth, 3, rate=1, data_format=data_format,
                            normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                            activation_fn=tf.nn.sigmoid,
                            weights_regularizer=regularizer,
                            weights_initializer=weights_initializer)
        # gate weights
        g = conv_mean_pool(full_inp, filter_depth, 3, rate=1, data_format=data_format,
                           normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                           activation_fn=activation_fn,
                           weights_regularizer=regularizer,
                           weights_initializer=weights_initializer)

        # bias variables
        bias_shape = g.get_shape().as_list()
        bias_shape[0] = 1
        bias_c = tf.get_variable("cell_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        bias_h = tf.get_variable("hidden_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        # new cell memory state
        ct_new = conv_mean_pool(ct, filter_depth, 1, data_format=data_format,
                                activation_fn=None, normalizer_fn=None,
                                weights_regularizer=regularizer, biases_initializer=None,
                                weights_initializer=weights_initializer) * fg + g * ig + bias_c
        # new hidden state
        ht_new = activation_fn(ct_new) * og + bias_h
    elif stride == 1:
        # input gate
        ig = ly.conv2d(full_inp, filter_depth, 3, stride=stride, rate=dilate, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # forget gate
        fg = ly.conv2d(full_inp, filter_depth, 3, stride=stride, rate=dilate, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # output gate
        og = ly.conv2d(full_inp, filter_depth, 3, stride=stride, rate=dilate, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # gate weights
        g = ly.conv2d(full_inp, filter_depth, 3, stride=stride, rate=dilate, data_format=data_format,
                      normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                      activation_fn=activation_fn,
                      weights_regularizer=regularizer,
                      weights_initializer=weights_initializer)

        # bias variables
        bias_shape = g.get_shape().as_list()
        bias_shape[0] = 1
        bias_c = tf.get_variable("cell_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        bias_h = tf.get_variable("hidden_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        # new cell memory state
        ct_new = ly.conv2d(ct, filter_depth, 1, stride=1, data_format=data_format,
                           activation_fn=None, normalizer_fn=None,
                           weights_regularizer=regularizer, biases_initializer=None,
                           weights_initializer=weights_initializer) * fg + g * ig + bias_c
        # new hidden state
        ht_new = activation_fn(ct_new) * og + bias_h
    else:
        raise NotImplementedError

    return ht_new, ct_new


def resnet_deconv_block(x, extra, filter_depth, stride,
                        norm_activate_first=False,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=ly.xavier_initializer_conv2d(),
                        data_format='NCHW',
                        weight_decay_rate=1e-8,
                        all_deconv=False):
    if all_deconv:
        followup_conv = ly.conv2d_transpose
    else:
        followup_conv = ly.conv2d

    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)
    in_filter_depth = int(x.get_shape()[channel_index])
    # assert in_filter_depth >= filter_depth

    orig_x = tf.identity(x)
    if norm_activate_first:
        with tf.variable_scope('residual_deconv_norm_activation'):
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                x = normalizer_fn(x, activation_fn=None, **normalizer_params)
            x = activation_fn(x)

    with tf.variable_scope('subs'):
        if extra is not None:
            x = tf.concat([x, extra], axis=channel_index)
        if stride == 2:
            x = upsample_conv(x, filter_depth, 3, data_format=data_format,
                              normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                              activation_fn=activation_fn,
                              weights_regularizer=regularizer,
                              weights_initializer=weights_initializer)
        elif stride == 1:
            x = ly.conv2d(x, filter_depth, 3, stride=stride, data_format=data_format,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          activation_fn=activation_fn, weights_regularizer=regularizer,
                          weights_initializer=weights_initializer)
        else:
            x = ly.conv2d_transpose(x, filter_depth, 3, stride=stride, data_format=data_format,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params,
                                    activation_fn=activation_fn, weights_regularizer=regularizer,
                                    weights_initializer=weights_initializer)
        x = followup_conv(x, filter_depth, 3, stride=1, data_format=data_format,
                          normalizer_fn=None,
                          activation_fn=None, weights_regularizer=regularizer,
                          weights_initializer=weights_initializer)

    with tf.variable_scope('sub_add'):
        if stride == 2:
            orig_x = upsample_conv(orig_x, filter_depth, 1, data_format=data_format,
                                   activation_fn=None, normalizer_fn=None,
                                   weights_regularizer=regularizer, weights_initializer=weights_initializer)
        elif stride == 1:
            orig_x = ly.conv2d(orig_x, filter_depth, 1, stride=stride, data_format=data_format,
                               activation_fn=None, normalizer_fn=None,
                               weights_regularizer=regularizer, weights_initializer=weights_initializer)
        else:
            orig_x = ly.conv2d_transpose(orig_x, filter_depth, 1, stride=stride, data_format=data_format,
                                         activation_fn=None, normalizer_fn=None,
                                         weights_regularizer=regularizer, weights_initializer=weights_initializer)
        x += orig_x

    # tf.logging.info('image after unit %s', x.get_shape())
    return x


def lstm_deconv_block(inp, ht, ct, filter_depth, stride,
                      norm_activate_first=None,
                      activation_fn=tf.nn.relu,
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_initializer=ly.xavier_initializer_conv2d(),
                      data_format='NCHW',
                      weight_decay_rate=1e-8,
                      all_deconv=None, norm_mask=False):
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)

    full_inp = tf.concat([inp, ht], axis=channel_index)
    if stride == 2:
        # input gate
        ig = upsample_conv(full_inp, filter_depth, 3, data_format=data_format,
                           normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                           activation_fn=tf.nn.sigmoid,
                           weights_regularizer=regularizer,
                           weights_initializer=weights_initializer)
        # forget gate
        fg = upsample_conv(full_inp, filter_depth, 3, data_format=data_format,
                           normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                           activation_fn=tf.nn.sigmoid,
                           weights_regularizer=regularizer,
                           weights_initializer=weights_initializer)
        # output gate
        og = upsample_conv(full_inp, filter_depth, 3, data_format=data_format,
                           normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                           activation_fn=tf.nn.sigmoid,
                           weights_regularizer=regularizer,
                           weights_initializer=weights_initializer)
        # gate weights
        g = upsample_conv(full_inp, filter_depth, 3, data_format=data_format,
                          normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                          activation_fn=activation_fn,
                          weights_regularizer=regularizer,
                          weights_initializer=weights_initializer)

        # bias variables
        bias_shape = g.get_shape().as_list()
        bias_shape[0] = 1
        bias_c = tf.get_variable("cell_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        bias_h = tf.get_variable("hidden_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        # new cell memory state
        ct_new = upsample_conv(ct, filter_depth, 1, data_format=data_format,
                               activation_fn=None, normalizer_fn=None,
                               weights_regularizer=regularizer, biases_initializer=None,
                               weights_initializer=weights_initializer) * fg + g * ig + bias_c
        # new hidden state
        ht_new = activation_fn(ct_new) * og + bias_h

        # # new cell memory state
        # ct_new = upsample_conv(ct, filter_depth, 3, data_format=data_format,
        #                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
        #                        activation_fn=activation_fn,
        #                        weights_regularizer=regularizer,
        #                        weights_initializer=weights_initializer) * fg + g * ig
        # # new hidden state
        # ht_new = ly.conv2d(ct_new, filter_depth, 3, data_format=data_format,
        #                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
        #                    activation_fn=activation_fn,
        #                    weights_regularizer=regularizer,
        #                    weights_initializer=weights_initializer) * og
    elif stride == 1:
        # input gate
        ig = ly.conv2d(full_inp, filter_depth, 3, stride=stride, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # forget gate
        fg = ly.conv2d(full_inp, filter_depth, 3, stride=stride, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # output gate
        og = ly.conv2d(full_inp, filter_depth, 3, stride=stride, data_format=data_format,
                       normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                       activation_fn=tf.nn.sigmoid,
                       weights_regularizer=regularizer,
                       weights_initializer=weights_initializer)
        # gate weights
        g = ly.conv2d(full_inp, filter_depth, 3, stride=stride, data_format=data_format,
                      normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                      activation_fn=activation_fn,
                      weights_regularizer=regularizer,
                      weights_initializer=weights_initializer)

        # bias variables
        bias_shape = g.get_shape().as_list()
        bias_shape[0] = 1
        bias_c = tf.get_variable("cell_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        bias_h = tf.get_variable("hidden_bias", shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        # new cell memory state
        ct_new = ly.conv2d(ct, filter_depth, 1, stride=1, data_format=data_format,
                           activation_fn=None, normalizer_fn=None,
                           weights_regularizer=regularizer, biases_initializer=None,
                           weights_initializer=weights_initializer) * fg + g * ig + bias_c
        # new hidden state
        ht_new = activation_fn(ct_new) * og + bias_h

        # # new cell memory state
        # ct_new = ly.conv2d(ct, filter_depth, 3, stride=stride, data_format=data_format,
        #                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
        #                    activation_fn=activation_fn,
        #                    weights_regularizer=regularizer,
        #                    weights_initializer=weights_initializer) * fg + g * ig
        # # new hidden state
        # ht_new = ly.conv2d(ct_new, filter_depth, 3, data_format=data_format,
        #                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
        #                    activation_fn=activation_fn,
        #                    weights_regularizer=regularizer,
        #                    weights_initializer=weights_initializer) * og
    else:
        raise NotImplementedError

    return ht_new, ct_new


def resnet_conv(x, filter_depth, stride=2, dilate_rate=1,
                num_blocks=5,
                first_unit=False, last_unit=False,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=ly.xavier_initializer_conv2d(),
                weight_decay_rate=1e-8,
                use_bottleneck=True, unit_num=0, data_format='NCHW'):
    # assert not (first_unit and last_unit)
    if dilate_rate != 1:
        stride = 1

    if use_bottleneck:
        raise NotImplementedError
    else:
        res_block = resnet_block

    with tf.variable_scope('resnet_unit_%d_0' % unit_num):
        x = res_block(x, filter_depth, stride=stride,
                      dilate=dilate_rate, first_layer_no_dilate=True,
                      norm_activate_first=(not first_unit),
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
                      normalizer_params=normalizer_params,
                      weights_initializer=weights_initializer,
                      weight_decay_rate=weight_decay_rate,
                      data_format=data_format)
    for j in range(1, num_blocks):
        with tf.variable_scope('resnet_unit_%d_%d' % (unit_num, j)):
            x = res_block(x, filter_depth, stride=1,
                          dilate=dilate_rate, first_layer_no_dilate=False,
                          norm_activate_first=True,
                          activation_fn=activation_fn,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          weights_initializer=weights_initializer,
                          weight_decay_rate=weight_decay_rate,
                          data_format=data_format)

    if last_unit:
        with tf.variable_scope('resnet_unit_%d_last' % unit_num):
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                x = normalizer_fn(x, activation_fn=None, **normalizer_params)
            x = activation_fn(x)

    return x


def resnet_deconv(x, extra, filter_depth, stride=2, num_blocks=5,
                  first_unit=False, last_unit=False,
                  activation_fn=tf.nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=ly.xavier_initializer_conv2d(),
                  weight_decay_rate=1e-8,
                  use_bottleneck=True, unit_num=0, data_format='NCHW'):
    # assert not (first_unit and last_unit)

    if use_bottleneck:
        raise NotImplementedError
    else:
        res_block = resnet_deconv_block

    with tf.variable_scope('resnet_deconv_unit_%d_0' % unit_num):
        x = res_block(x, extra, filter_depth, stride=stride, norm_activate_first=(not first_unit),
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
                      normalizer_params=normalizer_params,
                      weights_initializer=weights_initializer,
                      data_format=data_format,
                      weight_decay_rate=weight_decay_rate,
                      all_deconv=USE_ALL_DECONV)
    for j in range(1, num_blocks):
        with tf.variable_scope('resnet_deconv_unit_%d_%d' % (unit_num, j)):
            x = res_block(x, None, filter_depth, stride=1, norm_activate_first=True,
                          activation_fn=activation_fn,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          weights_initializer=weights_initializer,
                          data_format=data_format,
                          weight_decay_rate=weight_decay_rate,
                          all_deconv=USE_ALL_DECONV)

    if last_unit:
        with tf.variable_scope('resnet_deconv_unit_%d_last' % unit_num):
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                x = normalizer_fn(x, activation_fn=None, **normalizer_params)
            x = activation_fn(x)

    return x


def unrolled_lstm_conv(x, ht, ct, filter_depth, stride=2, dilate_rate=1,
                       num_blocks=5,
                       first_unit=False, last_unit=False,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=None,
                       normalizer_params=None,
                       weights_initializer=ly.xavier_initializer_conv2d(),
                       weight_decay_rate=1e-8,
                       use_bottleneck=None, unit_num=0, data_format='NCHW'):
    assert len(ht) == len(ct) and len(ct) == num_blocks

    if dilate_rate != 1:
        stride = 1

    if use_bottleneck:
        raise NotImplementedError
    else:
        res_block = lstm_conv_block

    hts_new = []
    cts_new = []
    inp = x
    with tf.variable_scope('lstm_conv_unit_t_%d_layer_0' % unit_num):
        ht_new, ct_new = res_block(inp, ht[0], ct[0], filter_depth, stride=stride,
                                   dilate=dilate_rate, first_layer_no_dilate=None,
                                   norm_activate_first=None,
                                   activation_fn=activation_fn,
                                   normalizer_fn=normalizer_fn,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=weights_initializer,
                                   data_format=data_format,
                                   weight_decay_rate=weight_decay_rate)
        hts_new.append(ht_new)
        cts_new.append(ct_new)
        inp = ht_new

    for i in range(1, num_blocks):
        with tf.variable_scope('lstm_deconv_unit_t_%d_layer_%d' % (unit_num, i)):
            ht_new, ct_new = res_block(inp, ht[i], ct[i], filter_depth, stride=1,
                                       dilate=dilate_rate, first_layer_no_dilate=None,
                                       norm_activate_first=None,
                                       activation_fn=activation_fn,
                                       normalizer_fn=normalizer_fn,
                                       normalizer_params=normalizer_params,
                                       weights_initializer=weights_initializer,
                                       data_format=data_format,
                                       weight_decay_rate=weight_decay_rate)
            hts_new.append(ht_new)
            cts_new.append(ct_new)
            inp = ht_new

    return hts_new, cts_new


def unrolled_lstm_deconv(x, ht, ct, filter_depth, stride=2, num_blocks=2,
                         first_unit=False, last_unit=False,
                         activation_fn=tf.nn.relu,
                         normalizer_fn=None,
                         normalizer_params=None,
                         weights_initializer=ly.xavier_initializer_conv2d(),
                         weight_decay_rate=1e-8,
                         use_bottleneck=None, unit_num=0, data_format='NCHW'):
    assert len(ht) == len(ct) and len(ct) == num_blocks

    if use_bottleneck:
        raise NotImplementedError
    else:
        res_block = lstm_deconv_block

    hts_new = []
    cts_new = []
    inp = x
    with tf.variable_scope('lstm_deconv_unit_t_%d_layer_0' % unit_num):
        ht_new, ct_new = res_block(inp, ht[0], ct[0], filter_depth, stride=stride, norm_activate_first=None,
                                   activation_fn=activation_fn,
                                   normalizer_fn=normalizer_fn,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=weights_initializer,
                                   data_format=data_format,
                                   weight_decay_rate=weight_decay_rate,
                                   all_deconv=USE_ALL_DECONV)
        hts_new.append(ht_new)
        cts_new.append(ct_new)
        inp = ht_new

    for i in range(1, num_blocks):
        with tf.variable_scope('lstm_deconv_unit_t_%d_layer_%d' % (unit_num, i)):
            ht_new, ct_new = res_block(inp, ht[i], ct[i], filter_depth, stride=1, norm_activate_first=None,
                                       activation_fn=activation_fn,
                                       normalizer_fn=normalizer_fn,
                                       normalizer_params=normalizer_params,
                                       weights_initializer=weights_initializer,
                                       data_format=data_format,
                                       weight_decay_rate=weight_decay_rate,
                                       all_deconv=USE_ALL_DECONV)
            hts_new.append(ht_new)
            cts_new.append(ct_new)
            inp = ht_new

    return hts_new, cts_new
