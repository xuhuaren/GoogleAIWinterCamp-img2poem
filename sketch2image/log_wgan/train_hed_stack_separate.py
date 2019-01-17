import os
from time import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from graph_supervised_wgan_crn_enc_rand_stack_gru import build_multi_tower_graph, build_single_graph_stage_1, build_single_graph_stage_2
from input_pipeline_rand_mix_stack import build_input_queue_paired_sketchy, build_input_queue_paired_sketchy_test, build_input_queue_paired_flickr, build_input_queue_paired_mixed
import inception_score

tf.logging.set_verbosity(tf.logging.INFO)
inception_v4_ckpt_path = './inception_v4_model/inception_v4.ckpt'
vgg_16_ckpt_path = './vgg_16_model/vgg_16.ckpt'


def one_hot_to_dense(labels):
    # Assume on value is 1
    batch_size = int(labels.get_shape()[0])
    return tf.reshape(tf.where(tf.equal(labels, 1))[:, 1], (batch_size,))


def print_parameter_count(verbose=False):
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s1'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('generator_s1')
    print(total_parameters)
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s2'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('generator_s2')
    print(total_parameters)
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('generator')
    print(total_parameters)

    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s1'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('critic_s1')
    print(total_parameters)

    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s2'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('critic_s2'
          '')
    print(total_parameters)
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('critic')
    print(total_parameters)


def train(**kwargs):

    def get_inception_score_origin(generator_out, data_format, session, n):
        all_samples = []
        img_dim = 64
        for i in range(n // 100):
            all_samples.append(session.run(generator_out[0]))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, img_dim, img_dim))
        if data_format == 'NCHW':
            all_samples = all_samples.transpose(0, 2, 3, 1)
        return inception_score.get_inception_score(list(all_samples), session)

    def get_inception_score(generator_out, batch_size, img_dim, channel, data_format, sess):
        all_samples = []
        for i in range(int(1000/batch_size)):
            all_samples.append(sess.run(generator_out))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        if data_format == 'NCHW':
            all_samples = all_samples.reshape((-1, channel, img_dim, img_dim)).transpose(0, 2, 3, 1)
        else:
            all_samples = all_samples.reshape((-1, img_dim, img_dim, channel))

        return inception_score.get_inception_score(list(all_samples), sess)

    resume = False
    iter_from = 0
    status = 0

    # Roll out the parameters
    appendix = kwargs["resume_from"]
    # dset1 = kwargs["dset1"]
    # dset2 = kwargs["dset2"]
    batch_size = kwargs["batch_size"]
    img_dim = kwargs["img_dim"]
    num_classes = kwargs["num_classes"]
    noise_dim = kwargs["noise_dim"]
    max_iter_step = kwargs["max_iter_step"]
    weight_decay_rate = kwargs["weight_decay_rate"]
    deconv_weight_decay_rate = kwargs["deconv_weight_decay_rate"]
    Diters = kwargs["disc_iterations"]
    ld = kwargs["lambda"]
    optimizer = kwargs["optimizer"]
    lr_G = kwargs["lr_G"]
    lr_D = kwargs["lr_D"]
    # device = kwargs["device"]
    num_gpu = kwargs["num_gpu"]
    log_dir = kwargs["log_dir"]
    ckpt_dir = kwargs["ckpt_dir"]
    data_format = kwargs["data_format"]
    distance_map = kwargs["distance_map"]
    small_img = kwargs["small_img"]

    if not (appendix is None or appendix == ''):
        resume = True
        iter_from = kwargs["iter_from"]

    # Temp - test auto method
    num_classes = None

    channel1 = 3
    channel2 = 3
    distance_map = distance_map != 0
    small = small_img != 0
    if small:
        img_dim = 64
    else:
        img_dim = 256
    batch_portion = np.array([1, 1, 1, 1], dtype=np.int32)

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")

    # Stage division
    mid_point = int(max_iter_step / 2)
    max_iter_step_s1 = mid_point
    max_iter_step_s2 = max_iter_step - mid_point

    stage_1_log_dir = os.path.join(log_dir, "stage1")
    if not os.path.exists(stage_1_log_dir):
        os.mkdir(stage_1_log_dir)
    stage_1_ckpt_dir = os.path.join(ckpt_dir, "stage1")
    if not os.path.exists(stage_1_ckpt_dir):
        os.mkdir(stage_1_ckpt_dir)
    stage_2_log_dir = os.path.join(log_dir, "stage2")
    if not os.path.exists(stage_2_log_dir):
        os.mkdir(stage_2_log_dir)
    stage_2_ckpt_dir = os.path.join(ckpt_dir, "stage2")
    if not os.path.exists(stage_2_ckpt_dir):
        os.mkdir(stage_2_ckpt_dir)

    #################################### Stage 1 ##################################
    if iter_from < mid_point:
        tf.reset_default_graph()
        print("Stage 1")
        print(iter_from)
        print(max_iter_step_s1)

        assert inception_score.softmax.graph != tf.get_default_graph()
        inception_score._init_inception()

        counter = tf.Variable(initial_value=iter_from, dtype=tf.int32, trainable=False)
        counter_addition_op = tf.assign_add(counter, 1, use_locking=True)
        portion = 0.1 + tf.minimum(0.8, (tf.cast(counter, tf.float32) / max_iter_step_s1 / 0.95) ** 1.0)

        # Construct data queue
        with tf.device('/cpu:0'):
            images_small, sketches_small, images_large, sketches_large, image_paired_class_ids = build_input_queue_paired_mixed(
                batch_size=batch_size * num_gpu,
                img_dim=img_dim,
                test_mode=False,
                # portion=tf.minimum(0.9, tf.cast(counter, tf.float32) / (0.9 * max_iter_step)),
                portion=portion,
                data_format=data_format,
                distance_map=distance_map,
                small=small, capacity=2 ** 12)
            image_paired_class_ids = one_hot_to_dense(image_paired_class_ids)
        with tf.device('/cpu:0'):
            images_small_d, _, _, _, _ = build_input_queue_paired_mixed(
                batch_size=batch_size * num_gpu,
                img_dim=img_dim,
                test_mode=False,
                # portion=tf.minimum(0.9, tf.cast(counter, tf.float32) / (0.9 * max_iter_step)),
                portion=tf.constant(0.3, dtype=tf.float32),
                data_format=data_format,
                distance_map=distance_map,
                small=small, capacity=2 ** 12)
            # image_paired_class_ids = one_hot_to_dense(image_paired_class_ids)
        with tf.device('/cpu:0'):
            _, sketches_small_100, _, sketches_large_100, image_paired_class_ids_100 = build_input_queue_paired_sketchy(
                batch_size=100,
                img_dim=img_dim,
                test_mode=False,
                data_format=data_format,
                distance_map=distance_map,
                small=small, capacity=1024)
            image_paired_class_ids_100 = one_hot_to_dense(image_paired_class_ids_100)

        opt_g, opt_d, loss_g, loss_d, merged_all, gen_out = build_multi_tower_graph(
            images_small, sketches_small, images_large, sketches_large,
            images_small_d,
            sketches_small_100, sketches_large_100,
            image_paired_class_ids, image_paired_class_ids_100,
            batch_size=batch_size, num_gpu=num_gpu, batch_portion=batch_portion, training=True,
            in_channel1=channel1, in_channel2=channel2, out_channel=channel1,
            img_dim=img_dim, num_classes=num_classes,
            learning_rates={
                "generator": lr_G,
                "discriminator": lr_D,
            },
            counter=counter, portion=portion, max_iter_step=max_iter_step_s1, stage=1,
            ld=ld, data_format=data_format,
            distance_map=distance_map,
            optimizer=optimizer)

        inception_score_mean = tf.placeholder(dtype=tf.float32, shape=())
        inception_score_std = tf.placeholder(dtype=tf.float32, shape=())
        inception_score_mean_summary = tf.summary.scalar("inception_score/mean", inception_score_mean)
        inception_score_std_summary = tf.summary.scalar("inception_score/std", inception_score_std)
        inception_score_summary = tf.summary.merge((inception_score_mean_summary, inception_score_std_summary))

        saver_s1 = tf.train.Saver()
        try:
            saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV4'))
            perceptual_model_path = inception_v4_ckpt_path
        except:
            try:
                saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16'))
                perceptual_model_path = vgg_16_ckpt_path
            except:
                saver2 = None

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                intra_op_parallelism_threads=10)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1   # JIT XLA
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if saver2 is not None:
                saver2.restore(sess, perceptual_model_path)

            # saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            summary_writer = tf.summary.FileWriter(stage_1_log_dir, sess.graph)
            if resume:
                saver_s1.restore(sess, tf.train.latest_checkpoint(stage_1_ckpt_dir))
                summary_writer.reopen()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            run_metadata = tf.RunMetadata()

            print_parameter_count(verbose=False)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run([counter.assign(iter_from)])

            for i in range(iter_from, max_iter_step_s1):
                if status == -1:
                    break

                if i % 100 == 0:
                    curr_time = time()
                    elapsed = curr_time - prev_time
                    print(
                        "Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter" % (i, elapsed, elapsed / 100.))
                    prev_time = curr_time

                diters = Diters

                # Train Discriminator
                for j in range(diters):
                    # print(j)
                    if i % 100 == 0 and j == 0:
                        _, merged, loss_d_out = sess.run([opt_d, merged_all, loss_d],
                                                         options=run_options,
                                                         run_metadata=run_metadata)
                        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        summary_writer.add_summary(merged, i)
                        summary_writer.add_run_metadata(
                            run_metadata, 'discriminator_metadata {}'.format(i), i)
                    else:
                        _, loss_d_out = sess.run([opt_d, loss_d])
                    if np.isnan(np.sum(loss_d_out)):
                        status = -1
                        print("NaN occurred during training D")
                        return status

                # Train Generator
                if i % 100 == 0:
                    _, merged, loss_g_out, counter_out, _ = sess.run(
                        [opt_g, merged_all, loss_g, counter, counter_addition_op],
                        options=run_options,
                        run_metadata=run_metadata)
                    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'generator_metadata {}'.format(i), i)
                else:
                    _, loss_g_out, counter_out, _ = sess.run([opt_g, loss_g, counter, counter_addition_op])
                if np.isnan(np.sum(loss_g_out)):
                    status = -1
                    print("NaN occurred during training G")
                    return status

                if i % 5000 == 4999:
                    saver_s1.save(sess, os.path.join(
                        stage_1_ckpt_dir, "model.ckpt"), global_step=i)

                if i % 1000 == 999:
                    # this_score = get_inception_score(gen_out[1], batch_size=batch_size, img_dim=img_dim, channel=3,
                    #                                  data_format=data_format, sess=sess)
                    this_score = get_inception_score_origin(gen_out, data_format=data_format,
                                                            session=sess, n=10000)
                    merged_sum = sess.run(inception_score_summary, feed_dict={
                        inception_score_mean: this_score[0],
                        inception_score_std: this_score[1],
                    })
                    summary_writer.add_summary(merged_sum, i)

            coord.request_stop()
            coord.join(threads)

    ################################### Stage 2 ######################################
    if iter_from < mid_point:
        iter_from = 0
    else:
        iter_from = max(0, iter_from - mid_point)
    tf.reset_default_graph()
    batch_size /= 2
    batch_size = int(batch_size)
    assert batch_size % 2 == 0
    print("Stage 2")
    print(iter_from)
    print(max_iter_step_s2)

    assert inception_score.softmax.graph != tf.get_default_graph()
    inception_score._init_inception()

    counter = tf.Variable(initial_value=iter_from, dtype=tf.int32, trainable=False)
    counter_addition_op = tf.assign_add(counter, 1, use_locking=True)
    portion = 0.1 + tf.minimum(0.75, (tf.cast(counter, tf.float32) / (0.9 * max_iter_step_s2)) ** 1.0)

    # Construct data queue
    with tf.device('/cpu:0'):
        images_small, sketches_small, images_large, sketches_large, image_paired_class_ids = build_input_queue_paired_mixed(
            batch_size=batch_size * num_gpu,
            img_dim=img_dim,
            test_mode=False,
            portion=portion,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=2 ** 12)
        image_paired_class_ids = one_hot_to_dense(image_paired_class_ids)
    with tf.device('/cpu:0'):
        _, sketches_small_100, _, sketches_large_100, image_paired_class_ids_100 = build_input_queue_paired_sketchy(
            batch_size=100,
            img_dim=img_dim,
            test_mode=False,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=1024)
        image_paired_class_ids_100 = one_hot_to_dense(image_paired_class_ids_100)

    opt_g, opt_d, loss_g, loss_d, merged_all, gen_out = build_multi_tower_graph(
        images_small, sketches_small, images_large, sketches_large,
        sketches_small_100, sketches_large_100,
        image_paired_class_ids, image_paired_class_ids_100,
        batch_size=batch_size, num_gpu=num_gpu, batch_portion=batch_portion, training=True,
        in_channel1=channel1, in_channel2=channel2, out_channel=channel1,
        img_dim=img_dim, num_classes=num_classes,
        learning_rates={
            "generator": lr_G,
            "discriminator": lr_D,
        },
        counter=counter, portion=portion, max_iter_step=max_iter_step_s2, stage=2,
        ld=ld, data_format=data_format,
        distance_map=distance_map,
        optimizer=optimizer)

    inception_score_mean = tf.placeholder(dtype=tf.float32, shape=())
    inception_score_std = tf.placeholder(dtype=tf.float32, shape=())
    inception_score_mean_summary = tf.summary.scalar("inception_score/mean", inception_score_mean)
    inception_score_std_summary = tf.summary.scalar("inception_score/std", inception_score_std)
    inception_score_summary = tf.summary.merge((inception_score_mean_summary, inception_score_std_summary))

    # Add stage 1 parameters
    var_collections = {
        'generator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s1'),
        'discriminator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s1'),
        'generator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s2'),
        'discriminator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s2'),
    }
    saver2 = None
    saver_s1 = tf.train.Saver(var_collections['generator_s1'])
    saver_s2 = tf.train.Saver()
    # try:
    #     saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV4'))
    #     perceptual_model_path = inception_v4_ckpt_path
    # except:
    #     try:
    #         saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16'))
    #         perceptual_model_path = vgg_16_ckpt_path
    #     except:
    #         saver2 = None

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                            intra_op_parallelism_threads=10)
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1   # JIT XLA
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # if saver2 is not None:
        #     saver2.restore(sess, perceptual_model_path)

        # saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        summary_writer = tf.summary.FileWriter(stage_2_log_dir, sess.graph)
        if resume:
            saver_s2.restore(sess, tf.train.latest_checkpoint(stage_2_ckpt_dir))
            summary_writer.reopen()
        else:
            saver_s1.restore(sess, tf.train.latest_checkpoint(stage_1_ckpt_dir))

        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        print_parameter_count(verbose=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(iter_from, max_iter_step_s2):
            if status == -1:
                break

            if i % 100 == 0:
                curr_time = time()
                elapsed = curr_time - prev_time
                print(
                    "Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter" % (
                    i, elapsed, elapsed / 100.))
                prev_time = curr_time

            diters = Diters

            # Train Discriminator
            for j in range(diters):
                # print(j)
                if i % 100 == 0 and j == 0:
                    _, merged, loss_d_out = sess.run([opt_d, merged_all, loss_d],
                                                     options=run_options,
                                                     run_metadata=run_metadata)
                    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'discriminator_metadata {}'.format(i), i)
                else:
                    _, loss_d_out = sess.run([opt_d, loss_d])
                if np.isnan(np.sum(loss_d_out)):
                    status = -1
                    print("NaN occurred during training D")
                    return status

            # Train Generator
            if i % 100 == 0:
                _, merged, loss_g_out, counter_out, _ = sess.run(
                    [opt_g, merged_all, loss_g, counter, counter_addition_op],
                    options=run_options,
                    run_metadata=run_metadata)
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
            else:
                _, loss_g_out, counter_out, _ = sess.run([opt_g, loss_g, counter, counter_addition_op])
            # print(counter_out)
            if np.isnan(np.sum(loss_g_out)):
                status = -1
                print("NaN occurred during training G")
                return status

            if i % 5000 == 4999:
                saver_s2.save(sess, os.path.join(
                    stage_2_ckpt_dir, "model.ckpt"), global_step=i)

            if i % 1000 == 999:
                # this_score = get_inception_score(gen_out[1], batch_size=batch_size, img_dim=img_dim, channel=3,
                #                                  data_format=data_format, sess=sess)
                this_score = get_inception_score_origin(gen_out, data_format=data_format,
                                                        session=sess, n=10000)
                merged_sum = sess.run(inception_score_summary, feed_dict={
                    inception_score_mean: this_score[0],
                    inception_score_std: this_score[1],
                })
                summary_writer.add_summary(merged_sum, i)

        coord.request_stop()
        coord.join(threads)

    return status


def test(**kwargs):

    def binarize(sketch, threshold=245):
        sketch[sketch < threshold] = 0
        sketch[sketch >= threshold] = 255
        return sketch

    from scipy import ndimage

    # Roll out the parameters
    appendix = kwargs["resume_from"]
    batch_size = kwargs["batch_size"]
    # img_dim = kwargs["img_dim"]
    num_classes = kwargs["num_classes"]
    # noise_dim = kwargs["noise_dim"]
    # max_iter_step = kwargs["max_iter_step"]
    # weight_decay_rate = kwargs["weight_decay_rate"]
    # deconv_weight_decay_rate = kwargs["deconv_weight_decay_rate"]
    # Diters = kwargs["disc_iterations"]
    # ld = kwargs["lambda"]
    # optimizer = kwargs["optimizer"]
    # lr_G = kwargs["lr_G"]
    # lr_D = kwargs["lr_D"]
    # num_gpu = kwargs["num_gpu"]
    log_dir = kwargs["log_dir"]
    ckpt_dir = kwargs["ckpt_dir"]
    data_format = kwargs["data_format"]
    distance_map = kwargs["distance_map"]
    small_img = kwargs["small_img"]
    # test_folder = kwargs["test_image_folder"]
    stage = kwargs["stage"]

    if stage == 1:
        build_func = build_single_graph_stage_1
    elif stage == 2:
        build_func = build_single_graph_stage_2
    else:
        raise ValueError
    channel = 3
    distance_map = distance_map != 0
    small = small_img != 0
    if small or stage == 1:
        img_dim = 64
    else:
        img_dim = 256
    # batch_size = 20
    # output_img = np.zeros((img_dim * 2, img_dim * batch_size, channel))
    output_folder = os.path.join(log_dir, 'out')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")
    # Construct data queue
    with tf.device('/cpu:0'):
        images_small, sketches_small, images_large, sketches_large, image_paired_class_ids, \
            categories, imagenet_ids, sketch_ids = build_input_queue_paired_sketchy_test(
                batch_size=batch_size,
                img_dim=img_dim,
                test_mode=True,
                data_format=data_format,
                distance_map=distance_map,
                small=small, capacity=512)
        image_paired_class_ids = one_hot_to_dense(image_paired_class_ids)

    with tf.device('/gpu:0'):
        ret_list = build_func(images_small, sketches_small, None, None, None, None,
                              image_paired_class_ids, None,
                              batch_size=batch_size, training=False,
                              in_channel1=channel, in_channel2=channel,
                              out_channel=channel,
                              img_dim=img_dim, num_classes=num_classes,
                              data_format=data_format,
                              distance_map=distance_map)

    if stage == 1:
        print("Stage 1")

        stage_1_log_dir = os.path.join(log_dir, "stage1")
        if not os.path.exists(stage_1_log_dir):
            raise RuntimeError
        stage_1_ckpt_dir = os.path.join(ckpt_dir, "stage1")
        if not os.path.exists(stage_1_ckpt_dir):
            raise RuntimeError

        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver.restore(sess, tf.train.latest_checkpoint(stage_1_ckpt_dir))
            counter = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            while True:
                try:
                    generated_img, gt_image, input_sketch, category, imagenet_id, sketch_id = sess.run(
                        [ret_list[0], ret_list[1], ret_list[2], categories, imagenet_ids, sketch_ids])
                except Exception as e:
                    print(e.args)
                    break

                if counter % 100 == 0:
                    curr_time = time()
                    elapsed = curr_time - prev_time
                    print(
                        "Now at iteration %d. Elapsed time: %.5fs." % (counter, elapsed))
                    prev_time = curr_time

                if data_format == 'NCHW':
                    generated_img = np.transpose(generated_img, (0, 2, 3, 1))
                    gt_image = np.transpose(gt_image, (0, 2, 3, 1))
                    input_sketch = np.transpose(input_sketch, (0, 2, 3, 1))
                generated_img = ((generated_img + 1) / 2.) * 255
                gt_image = ((gt_image + 1) / 2.) * 255
                input_sketch = ((input_sketch + 1) / 2.) * 255
                generated_img = generated_img[:, :, :, ::-1].astype(np.uint8)
                gt_image = gt_image[:, :, :, ::-1].astype(np.uint8)
                input_sketch = input_sketch.astype(np.uint8)

                # input_sketch = 1 - (input_sketch < 0.025)

                # for i in range(int(batch_size / 2)):
                #     output_img[:img_dim, i * img_dim:(i + 1) * img_dim, :] = input_sketch[i]
                #     output_img[img_dim:, i * img_dim:(i + 1) * img_dim, :] = generated_img[i]

                # output_img = output_img[:, :int(batch_size / 2 + 1) * img_dim, :]

                for i in range(batch_size):
                    this_prefix = '%s_%d_%d' % (category[i].decode('ascii'),
                                                int(imagenet_id[i].decode('ascii').split('_')[1]),
                                                sketch_id[i])
                    img_out_filename = this_prefix + '_fake_B.png'
                    img_gt_filename = this_prefix + '_real_B.png'
                    sketch_in_filename = this_prefix + '_real_A.png'

                    # Save file
                    # file_path = os.path.join(output_folder, 'output_%d.jpg' % int(counter / batch_size))
                    cv2.imwrite(os.path.join(output_folder, img_out_filename), generated_img[i])
                    cv2.imwrite(os.path.join(output_folder, img_gt_filename), gt_image[i])
                    cv2.imwrite(os.path.join(output_folder, sketch_in_filename), input_sketch[i])
                    # output_img = np.zeros((img_dim * 2, img_dim * batch_size, channel))

                    print('Saved file %s' % this_prefix)

                    counter += 1

            coord.request_stop()
            coord.join(threads)
    else:
        raise NotImplementedError
