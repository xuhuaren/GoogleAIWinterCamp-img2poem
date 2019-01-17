import time
import os

import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

import models_crn_gan_enc_stack_gru_small as models
from input_pipeline_rand_mix_stack import *
from resnet_crn import mean_pool, upsample
from inception_v4 import inception_v4_base, inception_v4, inception_v4_arg_scope
from vgg import vgg_16, vgg_19, vgg_arg_scope
# from vgg16 import Vgg16


slim = tf.contrib.slim


CRAMER = False


def dist_map_to_image(input, threshold=0.025):
    ret = tf.cast(1 - tf.cast(tf.less(input + 1, threshold), tf.int32), tf.float32)
    # ret = tf.cast(1 - tf.cast(tf.equal(input, -1), tf.int32), tf.float32)
    # ret = tf.Print(ret, [input + 1, tf.equal(input + 1, 0), tf.reduce_sum(tf.cast(tf.less(input + 1, threshold), tf.float32))],
    #                first_n=1, summarize=20000)
    # ret = tf.Print(ret, [tf.reduce_min(input), tf.reduce_max(input), tf.reduce_min(ret), tf.reduce_max(ret)])
    # ret = tf.Print(ret, [tf.reduce_sum(tf.cast(tf.equal(input, 0), tf.float32))])
    return ret


def compute_gradients(losses, optimizers, var_lists):
    assert len(losses) == len(optimizers) and len(optimizers) == len(var_lists)
    grads = []
    for i in range(len(losses)):
        this_grad = optimizers[i].compute_gradients(losses[i], var_list=var_lists[i])
        grads.append(this_grad)
    return grads


def average_gradients(tower_grads_list):
    """notice: Variable pointers come from the first tower"""

    grads_list = []
    for i in range(len(tower_grads_list)):
        average_grads = []
        tower_grads = tower_grads_list[i]
        num_towers = len(tower_grads)
        for grad_and_vars in zip(*tower_grads):
            grads = []
            grad = 'No Value'
            if grad_and_vars[0][0] is None:
                all_none = True
                for j in range(num_towers):
                    if grad_and_vars[j][0] is not None:
                        all_none = False
                if all_none:
                    grad = None
                else:
                    raise ValueError("None gradient inconsistent between towers.")
            else:
                for g, _ in grad_and_vars:
                    expanded_grad = tf.expand_dims(g, axis=0)
                    grads.append(expanded_grad)

                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]
            if isinstance(grad, str):
                raise ValueError("Gradient not defined when averaging.")
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        grads_list.append(average_grads)
    return grads_list


def gather_summaries(max_outputs=100):
    # Image summaries
    orig_img_sum1 = tf.summary.image("original_img_1", tf.get_collection("original_img_1")[0], max_outputs=max_outputs)
    orig_img_sum1 = tf.summary.image("original_img_1_d", tf.get_collection("original_img_1_d")[0], max_outputs=max_outputs)
    orig_img_sum1 = tf.summary.image("original_img_1_l", tf.get_collection("original_img_1_l")[0], max_outputs=max_outputs)
    orig_img_sum2 = tf.summary.image("original_img_2", tf.get_collection("original_img_2")[0], max_outputs=max_outputs)
    orig_img_sum2 = tf.summary.image("original_img_2_l", tf.get_collection("original_img_2_l")[0], max_outputs=max_outputs)
    img_sum_2t1 = tf.summary.image("img_2_to_1_s1", tf.get_collection("img_2_to_1_s1")[0], max_outputs=max_outputs)
    img_sum_2t1 = tf.summary.image("img_2_to_1_s2", tf.get_collection("img_2_to_1_s2")[0], max_outputs=max_outputs)
    img_sum_2t1 = tf.summary.image("img_2_to_1_s1_b", tf.get_collection("img_2_to_1_s1_b")[0], max_outputs=max_outputs)
    # img_sum_212 = tf.summary.image("img_212", tf.get_collection("img_212")[0], max_outputs=max_outputs)
    if len(tf.get_collection("dist_map_img_2")) > 0:
        # dist_map_sum_1t2 = tf.summary.image("dist_map_1_to_2", tf.get_collection("dist_map_1_to_2")[0], max_outputs=max_outputs)
        dist_map_sum_2 = tf.summary.image("dist_map_img_2", tf.get_collection("dist_map_img_2")[0], max_outputs=max_outputs)
        # dist_map_sum_212 = tf.summary.image("dist_map_212", tf.get_collection("dist_map_212")[0], max_outputs=max_outputs)

    # # Intermediate layer
    # img_inter_1_1 = tf.summary.image("inception_layer_1_1",
    #                                  tf.concat(tf.get_collection("inception_layer_1_1"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_1_2 = tf.summary.image("inception_layer_1_2",
    #                                  tf.concat(tf.get_collection("inception_layer_1_2"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_1_3 = tf.summary.image("inception_layer_1_3",
    #                                  tf.concat(tf.get_collection("inception_layer_1_3"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_1_4 = tf.summary.image("inception_layer_1_4",
    #                                  tf.concat(tf.get_collection("inception_layer_1_4"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_1_5 = tf.summary.image("inception_layer_1_5",
    #                                  tf.concat(tf.get_collection("inception_layer_1_5"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_2_1 = tf.summary.image("inception_layer_2_1",
    #                                  tf.concat(tf.get_collection("inception_layer_2_1"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_2_2 = tf.summary.image("inception_layer_2_2",
    #                                  tf.concat(tf.get_collection("inception_layer_2_2"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_2_3 = tf.summary.image("inception_layer_2_3",
    #                                  tf.concat(tf.get_collection("inception_layer_2_3"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_2_4 = tf.summary.image("inception_layer_2_4",
    #                                  tf.concat(tf.get_collection("inception_layer_2_4"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)
    # img_inter_2_5 = tf.summary.image("inception_layer_2_5",
    #                                  tf.concat(tf.get_collection("inception_layer_2_5"), axis=0)[:, :, :, :3],
    #                                  max_outputs=max_outputs)

    # Scalar
    # tf.summary.scalar("DRAGAN_loss_g/2t1", tf.reduce_mean(tf.get_collection("DRAGAN_loss_g_2t1")))
    # tf.summary.scalar("DRAGAN_loss_d/2t1", tf.reduce_mean(tf.get_collection("DRAGAN_loss_d_2t1")))
    # tf.summary.scalar("DRAGAN_loss_g/1t2", tf.reduce_mean(tf.get_collection("DRAGAN_loss_g_1t2")))
    # tf.summary.scalar("DRAGAN_loss_d/1t2", tf.reduce_mean(tf.get_collection("DRAGAN_loss_d_1t2")))
    tf.summary.scalar("DRAGAN_loss_g/stage1", tf.reduce_mean(tf.get_collection("DRAGAN_loss_g_s1")))
    tf.summary.scalar("DRAGAN_loss_d/stage1", tf.reduce_mean(tf.get_collection("DRAGAN_loss_d_s1")))
    tf.summary.scalar("DRAGAN_loss_g/stage2", tf.reduce_mean(tf.get_collection("DRAGAN_loss_g_s2")))
    tf.summary.scalar("DRAGAN_loss_d/stage2", tf.reduce_mean(tf.get_collection("DRAGAN_loss_d_s2")))
    tf.summary.scalar("DRAGAN_loss_g/total", tf.reduce_mean(tf.get_collection("DRAGAN_loss_g")))
    tf.summary.scalar("DRAGAN_loss_d/total", tf.reduce_mean(tf.get_collection("DRAGAN_loss_d")))

    tf.summary.scalar("ACGAN_loss_g/s1", tf.reduce_mean(tf.get_collection("ACGAN_loss_g_s1")))
    tf.summary.scalar("ACGAN_loss_d/s1", tf.reduce_mean(tf.get_collection("ACGAN_loss_d_s1")))
    tf.summary.scalar("ACGAN_loss_g/s2", tf.reduce_mean(tf.get_collection("ACGAN_loss_g_s2")))
    tf.summary.scalar("ACGAN_loss_d/s2", tf.reduce_mean(tf.get_collection("ACGAN_loss_d_s2")))
    tf.summary.scalar("ACGAN_loss_g/total", tf.reduce_mean(tf.get_collection("ACGAN_loss_g")))
    tf.summary.scalar("ACGAN_loss_d/total", tf.reduce_mean(tf.get_collection("ACGAN_loss_d")))

    tf.summary.scalar("direct_loss_s1", tf.reduce_mean(tf.get_collection("direct_loss_s1")))
    tf.summary.scalar("direct_loss_s2", tf.reduce_mean(tf.get_collection("direct_loss_s2")))
    tf.summary.scalar("direct_loss", tf.reduce_mean(tf.get_collection("direct_loss")))
    tf.summary.scalar("fisher_alpha", tf.reduce_mean(tf.get_collection("fisher_alpha")))
    tf.summary.scalar("WGAN_loss/d", tf.reduce_mean(tf.get_collection("WGAN_loss_d")))
    tf.summary.scalar("WGAN_loss/gp", tf.reduce_mean(tf.get_collection("WGAN_loss_gp")))
    tf.summary.scalar("diverse_loss", tf.reduce_mean(tf.get_collection("diverse_loss")))
    tf.summary.scalar("regularization_loss", tf.reduce_mean(tf.get_collection("regularization_loss")))
    tf.summary.scalar("sketch_portion", tf.reduce_mean(tf.get_collection("sketch_portion")))

    tf.summary.scalar("total_loss/g", tf.reduce_mean(tf.get_collection("total_loss_g")))
    tf.summary.scalar("total_loss/d", tf.reduce_mean(tf.get_collection("total_loss_d")))

    return tf.summary.merge_all()


def gather_losses():
    loss_g = tf.reduce_mean(tf.get_collection("loss_g"))
    loss_d = tf.reduce_mean(tf.get_collection("loss_d"))
    # loss_su_g = tf.reduce_mean(tf.get_collection("loss_su_g"))
    return loss_g, loss_d


def build_multi_tower_graph(images_small, sketches_small, images_large, sketches_large,
                            images_small_d,
                            sketches_small_100, sketches_large_100,
                            image_paired_class_ids, image_paired_class_ids_100,
                            batch_size, num_gpu, batch_portion, training,
                            in_channel1, in_channel2, out_channel,
                            img_dim, num_classes,
                            learning_rates, counter, portion,
                            max_iter_step, stage=None,
                            ld=10,
                            data_format='NCHW', distance_map=True,
                            optimizer='Adam', **kwargs):
    assert stage is not None
    models.set_param(data_format=data_format)
    tf.add_to_collection("sketch_portion", portion)

    with tf.device('/cpu:0'):
        images_small_list = split_inputs(images_small, batch_size, batch_portion, num_gpu)
        images_small_d_list = split_inputs(images_small_d, batch_size, batch_portion, num_gpu)
        sketches_small_list = split_inputs(sketches_small, batch_size, batch_portion, num_gpu)
        images_large_list = split_inputs(images_large, batch_size, batch_portion, num_gpu)
        sketches_large_list = split_inputs(sketches_large, batch_size, batch_portion, num_gpu)
        image_paired_class_ids_list = split_inputs(image_paired_class_ids, batch_size, batch_portion, num_gpu)
        sketches_small_100_list = [tf.identity(sketches_small_100)] * len(batch_portion)
        sketches_large_100_list = [tf.identity(sketches_large_100)] * len(batch_portion)
        image_paired_class_ids_100_list = [tf.identity(image_paired_class_ids_100)] * len(batch_portion)

    lr_g = learning_rates['generator']
    lr_d = learning_rates['discriminator']
    # counter_g = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
    # counter_d = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
    optimizer = get_optimizer(optimizer)
    decay = tf.maximum(0.2, 1. - (tf.cast(counter, tf.float32) / max_iter_step * 0.9))
    optim_g = optimizer(learning_rate=lr_g * decay)
    optim_d = optimizer(learning_rate=lr_d * decay)

    # stepsize = 5e3
    # min_lr = 5e-5
    # max_lr = 1e-3
    # lr_g = tf.abs(tf.mod(tf.cast(counter, tf.float32), stepsize) - stepsize / 2.) / stepsize * 2. * (
    #     max_lr - min_lr) + min_lr
    # lr_d = tf.abs(tf.mod(tf.cast(counter, tf.float32), stepsize) - stepsize / 2.) / stepsize * 2. * (
    #     max_lr - min_lr) + min_lr
    # optim_g = optimizer(learning_rate=lr_g)
    # optim_d = optimizer(learning_rate=lr_d)

    tower_grads_g = []
    tower_grads_d = []
    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
                if stage == 1:
                    loss_g, loss_d, grad_g, grad_d, \
                        gen_out = build_single_graph_stage_1(images_small_list[i],
                                                             sketches_small_list[i],
                                                             images_large_list[i],
                                                             sketches_large_list[i],
                                                             images_small_d_list[i],
                                                             sketches_small_100_list[i],
                                                             image_paired_class_ids_list[i],
                                                             image_paired_class_ids_100_list[i],
                                                             batch_size * batch_portion[i],
                                                             training,
                                                             in_channel1, in_channel2, out_channel,
                                                             img_dim, num_classes,
                                                             ld=ld, data_format=data_format,
                                                             distance_map=distance_map,
                                                             optim_g=optim_g,
                                                             optim_d=optim_d)
                elif stage == 2:
                    loss_g, loss_d, grad_g, grad_d, \
                        gen_out = build_single_graph_stage_2(images_small_list[i],
                                                             sketches_small_list[i],
                                                             images_large_list[i],
                                                             sketches_large_list[i],
                                                             sketches_small_100_list[i],
                                                             sketches_large_100_list[i],
                                                             image_paired_class_ids_list[i],
                                                             image_paired_class_ids_100_list[i],
                                                             batch_size * batch_portion[i],
                                                             training,
                                                             in_channel1, in_channel2, out_channel,
                                                             img_dim, num_classes,
                                                             ld=ld, data_format=data_format,
                                                             distance_map=distance_map,
                                                             optim_g=optim_g,
                                                             optim_d=optim_d)
                else:
                    raise ValueError

                tower_grads_g.append(grad_g)
                tower_grads_d.append(grad_d)

    assert len(tower_grads_g) == len(tower_grads_d)
    if len(tower_grads_d) == 1:
        ave_grad_g = grad_g
        ave_grad_d = grad_d
    else:
        ave_grad_g, ave_grad_d = average_gradients((tower_grads_g, tower_grads_d))

    # Apply gradients
    tf.get_variable_scope()._reuse = False    # Hack to force initialization of optimizer variables

    # Clip gradients for MRU D
    if stage == 1:
        max_grad_norm_G = 50
        max_grad_norm_D = 100
        hard_clip_norm_G = 5
        hard_clip_norm_D = 10
        global_grad_norm_G = None
        global_grad_norm_G_clipped = None
        global_grad_norm_D = None
        global_grad_norm_D_clipped = None
    elif stage == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError
    ave_grad_g_tensors, ave_grad_g_vars = list(zip(*ave_grad_g))
    global_grad_norm_G = clip_ops.global_norm(ave_grad_g_tensors)
    ave_grad_g_tensors, _ = clip_ops.clip_by_global_norm(ave_grad_g_tensors, max_grad_norm_G, global_grad_norm_G)
    ave_grad_g_tensors = [clip_ops.clip_by_norm(t, hard_clip_norm_G) for t in ave_grad_g_tensors]
    ave_grad_g = list(zip(ave_grad_g_tensors, ave_grad_g_vars))

    ave_grad_d_tensors, ave_grad_d_vars = list(zip(*ave_grad_d))
    global_grad_norm_D = clip_ops.global_norm(ave_grad_d_tensors)
    ave_grad_d_tensors, _ = clip_ops.clip_by_global_norm(ave_grad_d_tensors, max_grad_norm_D, global_grad_norm_D)
    ave_grad_d_tensors = [clip_ops.clip_by_norm(t, hard_clip_norm_D) for t in ave_grad_d_tensors]
    ave_grad_d = list(zip(ave_grad_d_tensors, ave_grad_d_vars))
    opt_g = optimize(ave_grad_g, optim_g, None, 'gradient_norm', global_norm=global_grad_norm_G,
                     global_norm_clipped=global_grad_norm_G_clipped, appendix='_G')
    opt_d = optimize(ave_grad_d, optim_d, None, 'gradient_norm', global_norm=global_grad_norm_D,
                     global_norm_clipped=global_grad_norm_D_clipped, appendix='_D')

    summaries = gather_summaries()
    loss_g, loss_d = gather_losses()
    # loss_g = tf.Print(loss_g, [counter, lr_g])

    # Generator output from last tower
    return opt_g, opt_d, loss_g, loss_d, summaries, gen_out


def build_single_graph_stage_1(image_small, sketches_small, image_large, sketches_large,
                               image_small_d,
                               sketches_small_100,
                               image_data_class_id, image_data_2_class_id_100,
                               batch_size, training,
                               in_channel1, in_channel2, out_channel,
                               img_dim, num_classes,
                               ld=10,
                               data_format='NCHW', distance_map=True,
                               optim_g=None, optim_d=None):

    def perturb(input_data):
        input_dims = len(input_data.get_shape())
        reduce_axes = [0] + list(range(1, input_dims))
        ret = input_data + 0.5 * tf.sqrt(tf.nn.moments(input_data, axes=reduce_axes)[1]) * tf.random_uniform(input_data.shape)
        # ret = input_data + tf.random_normal(input_data.shape, stddev=2.0)
        return ret

    def transfer(image_data_list, labels_list, num_classes, reuse=False, data_format=data_format,
                 output_channel=3, stage=None, count=3):
        if stage == 1:
            generator = generator_s1
            generator_scope = 'generator_s1'
        elif stage == 2:
            generator = generator_s2
            generator_scope = 'generator_s2'
        else:
            raise ValueError('Invalid transfer stage')

        image_gens = []
        image_labels = []

        for i in range(count):
            image_gen = generator(image_data_list[i], output_channel=output_channel, num_classes=num_classes,
                                  reuse=reuse, data_format=data_format, labels=labels_list[i],
                                  scope_name=generator_scope)
            image_gens.append(image_gen)
            image_labels.append(labels_list[i])
            reuse = True

        return image_gens, image_labels

    models.set_param(data_format=data_format)
    num_classes = get_num_classes()

    ############################# Graph #################################
    # Input
    generator_s1 = models.generator_s1
    generator_s2 = models.generator_s2
    discriminator_s1 = models.critic_s1
    discriminator_s2 = models.critic_s2
    if CRAMER:
        discriminator_s1 = functools.partial(discriminator_s1, cramer=True)
        discriminator_s2 = functools.partial(discriminator_s2, cramer=True)
    # discriminator_cond = models.critic_cond
    # discriminator_e = models.critic_e
    # image_encoder = models.image_encoder
    # vae_sampler = models.vae_sampler

    # assert batch_size > 4 and batch_size % 4 == 0

    if image_large is None and sketches_large is None:
        image_small_list = tf.split(image_small, 1, axis=0)
        sketches_small_list = tf.split(sketches_small, 1, axis=0)
        image_data_class_id_list = tf.split(image_data_class_id, 1, axis=0)
    elif image_small is None and sketches_small is None:
        image_large_list = tf.split(image_large, 1, axis=0)
        sketches_large_list = tf.split(sketches_large, 1, axis=0)
        image_data_class_id_list = tf.split(image_data_class_id, 1, axis=0)
    else:
        image_small_list = tf.split(image_small, 2, axis=0)
        image_small_d_list = tf.split(image_small_d, 2, axis=0)
        sketches_small_list = tf.split(sketches_small, 2, axis=0)
        image_large_list = tf.split(image_large, 2, axis=0)
        sketches_large_list = tf.split(sketches_large, 2, axis=0)
        image_data_class_id_list = tf.split(image_data_class_id, 2, axis=0)

    image_gens_s1, image_labels_s1 = transfer(sketches_small_list, image_data_class_id_list,
                                              num_classes=num_classes, reuse=False, data_format=data_format,
                                              output_channel=3, stage=1, count=1)
    image_gens_s1_b, image_labels_s1_b = transfer(sketches_small_list, image_data_class_id_list,
                                                  num_classes=num_classes, reuse=True, data_format=data_format,
                                                  output_channel=3, stage=1, count=1)

    if not training:
        return image_gens_s1[0][-1], image_small, sketches_small

    # Inception Generation
    image_gen_100_s1, _ = transfer([sketches_small_100], [image_data_2_class_id_100], num_classes=num_classes,
                                   reuse=True, output_channel=3, data_format=data_format, stage=1, count=1)

    # Discriminator
    # Stage 1
    real_disc_out_s1, real_logit_s1 = discriminator_s1(image_small_d_list[1],
                                                       num_classes=num_classes, reuse=False, data_format=data_format,
                                                       scope_name='discriminator_s1')
    fake_disc_outs_s1 = []
    fake_logits_s1 = []
    for i in range(1):
        fake_disc_out_s1, fake_logit_s1 = discriminator_s1(image_gens_s1[i],
                                                           num_classes=num_classes, reuse=True,
                                                           data_format=data_format, scope_name='discriminator_s1')
        fake_disc_outs_s1.append(fake_disc_out_s1)
        fake_logits_s1.append(fake_logit_s1)
    ############################# End Graph ##############################

    loss_g_s1, loss_d_s1 = get_losses(discriminator_s1, discriminator_s2, None, None,
                                      batch_size, num_classes, img_dim, out_channel, 1,
                                      data_format, ld, perturb,
                                      # images
                                      image_small_list, sketches_small_list,
                                      image_large_list, sketches_large_list,
                                      image_small_d_list,
                                      image_gens_s1, image_gens_s1_b, None,
                                      # latent and labels
                                      image_data_class_id_list,
                                      image_labels_s1, image_labels_s1_b, None,
                                      # critic out
                                      real_disc_out_s1, fake_disc_outs_s1,
                                      None, None,
                                      # logit
                                      real_logit_s1, fake_logits_s1,
                                      None, None,
                                      )

    if data_format == 'NCHW':
        tf.add_to_collection("original_img_1", tf.transpose(tf.concat(image_small_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_1_l", tf.transpose(tf.concat(image_large_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_2", tf.transpose(tf.concat(sketches_small_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_2_l", tf.transpose(tf.concat(sketches_large_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_s1", tf.transpose(tf.concat([t[-1] for t in image_gens_s1], axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_s2", tf.zeros_like(tf.transpose(tf.concat([t[-1] for t in image_gens_s1], axis=0), (0, 2, 3, 1))))
        tf.add_to_collection("img_2_to_1_s1_b", tf.transpose(tf.concat([t[-1] for t in image_gens_s1_b], axis=0), (0, 2, 3, 1)))
        if distance_map:
            # tf.add_to_collection("dist_map_1_to_2",
            #                      tf.transpose(dist_map_to_image(tf.concat(image_gens_1t2, axis=0)), (0, 2, 3, 1)))
            tf.add_to_collection("dist_map_img_2",
                                 tf.transpose(dist_map_to_image(tf.concat(sketches_small_list, axis=0), threshold=0.025), (0, 2, 3, 1)))
            # tf.add_to_collection("dist_map_212",
            #                      tf.transpose(dist_map_to_image(tf.concat(image_gens_212, axis=0)), (0, 2, 3, 1)))

    # Add loss to collections
    tf.add_to_collection("loss_g", loss_g_s1)
    tf.add_to_collection("loss_d", loss_d_s1)

    # Variable Collections
    var_collections = {
        'generator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s1'),
        'discriminator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s1'),
        'generator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s2'),
        'discriminator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s2'),
        'generator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
        'discriminator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
    }

    ############# Reuse Variables for next tower (?) #############
    tf.get_variable_scope().reuse_variables()
    ############# Reuse Variables for next tower (?) #############

    # # Gather summaries from last tower
    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

    # Calculate Gradient
    grad_g_s1, grad_d_s1 = compute_gradients((loss_g_s1, loss_d_s1),
                                             (optim_g, optim_d),
                                             var_lists=(var_collections['generator_s1'],
                                                        var_collections['discriminator_s1']))

    # return opt_g, opt_d, opt_su_g, loss_g, loss_d, loss_su_g
    return loss_g_s1, loss_d_s1, grad_g_s1, grad_d_s1, (image_gen_100_s1[0],)


def build_single_graph_stage_2(image_small, sketches_small, image_large, sketches_large,
                               sketches_small_100, sketches_large_100,
                               image_data_class_id, image_data_2_class_id_100,
                               batch_size, training,
                               in_channel1, in_channel2, out_channel,
                               img_dim, num_classes,
                               ld=10,
                               data_format='NCHW', distance_map=True,
                               optim_g=None, optim_d=None):

    def perturb(input_data):
        input_dims = len(input_data.get_shape())
        reduce_axes = [0] + list(range(1, input_dims))
        ret = input_data + 0.5 * tf.sqrt(tf.nn.moments(input_data, axes=reduce_axes)[1]) * tf.random_uniform(input_data.shape)
        # ret = input_data + tf.random_normal(input_data.shape, stddev=2.0)
        return ret

    def transfer(input_data_list, extra_input_list, labels_list, num_classes, reuse=False, data_format=data_format,
                 output_channel=3, stage=None, count=3):
        if stage == 1:
            generator = generator_s1
            generator_scope = 'generator_s1'
        elif stage == 2:
            generator = generator_s2
            generator_scope = 'generator_s2'
        else:
            raise ValueError('Invalid transfer stage')

        image_gens = []
        image_labels = []

        for i in range(count):
            if stage == 1:
                image_gen = generator(input_data_list[i], output_channel=output_channel, num_classes=num_classes,
                                      reuse=reuse, data_format=data_format, labels=labels_list[i],
                                      scope_name=generator_scope)
            elif stage == 2:
                image_gen = generator(input_data_list[i], extra_input_list[i], output_channel=output_channel,
                                      num_classes=num_classes, reuse=reuse, data_format=data_format,
                                      labels=labels_list[i], scope_name=generator_scope)
            image_gens.append(image_gen)
            image_labels.append(labels_list[i])
            reuse = True

        return image_gens, image_labels

    models.set_param(data_format=data_format)
    num_classes = get_num_classes()

    ############################# Graph #################################
    # Input
    generator_s1 = models.generator_s1
    generator_s2 = models.generator_s2
    discriminator_s1 = models.critic_s1
    discriminator_s2 = models.critic_s2
    if CRAMER:
        discriminator_s1 = functools.partial(discriminator_s1, cramer=True)
        discriminator_s2 = functools.partial(discriminator_s2, cramer=True)
    # discriminator_cond = models.critic_cond
    # discriminator_e = models.critic_e
    # image_encoder = models.image_encoder
    # vae_sampler = models.vae_sampler

    # assert batch_size > 4 and batch_size % 4 == 0

    image_small_list = tf.split(image_small, 2, axis=0)
    sketches_small_list = tf.split(sketches_small, 2, axis=0)
    image_large_list = tf.split(image_large, 2, axis=0)
    sketches_large_list = tf.split(sketches_large, 2, axis=0)
    image_data_class_id_list = tf.split(image_data_class_id, 2, axis=0)

    image_gens_s1, image_labels_s1 = transfer(sketches_small_list, None, image_data_class_id_list,
                                              num_classes=num_classes, reuse=False, data_format=data_format,
                                              output_channel=3, stage=1, count=1)
    image_gens_s2, image_labels_s2 = transfer(image_gens_s1, sketches_large_list, image_data_class_id_list,
                                              num_classes=num_classes, reuse=False, data_format=data_format,
                                              output_channel=3, stage=2, count=1)

    if not training:
        return image_gens_s1[0], image_gens_s2[0], image_small, sketches_small

    # Inception Generation
    image_gen_100_s1, _ = transfer([sketches_small_100], None, [image_data_2_class_id_100], num_classes=num_classes,
                                   reuse=True, output_channel=3, data_format=data_format, stage=1, count=1)
    image_gen_100_s2, _ = transfer(image_gen_100_s1, [sketches_large_100], [image_data_2_class_id_100], num_classes=num_classes,
                                   reuse=True, output_channel=3, data_format=data_format, stage=2, count=1)

    # Discriminator
    # Stage 2
    real_disc_out_s2, real_logit_s2 = discriminator_s2(image_large_list[1],
                                                       num_classes=num_classes, reuse=False, data_format=data_format,
                                                       scope_name='discriminator_s2')
    fake_disc_outs_s2 = []
    fake_logits_s2 = []
    for i in range(1):
        fake_disc_out_s2, fake_logit_s2 = discriminator_s2(image_gens_s2[i],
                                                           num_classes=num_classes, reuse=True,
                                                           data_format=data_format, scope_name='discriminator_s2')
        fake_disc_outs_s2.append(fake_disc_out_s2)
        fake_logits_s2.append(fake_logit_s2)
    ############################# End Graph ##############################

    loss_g_s2, loss_d_s2 = get_losses(discriminator_s1, discriminator_s2, None, None,
                                      batch_size, num_classes, img_dim, out_channel, 2,
                                      data_format, ld, perturb,
                                      # images
                                      image_small_list, sketches_small_list,
                                      image_large_list, sketches_large_list,
                                      image_gens_s1, None, image_gens_s2,
                                      # latent and labels
                                      image_data_class_id_list,
                                      image_labels_s1, None, image_labels_s2,
                                      # critic out
                                      None, None,
                                      real_disc_out_s2, fake_disc_outs_s2,
                                      # logit
                                      None, None,
                                      real_logit_s2, fake_logits_s2,
                                      )

    if data_format == 'NCHW':
        tf.add_to_collection("original_img_1", tf.transpose(tf.concat(image_small_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_1_l", tf.transpose(tf.concat(image_large_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_2", tf.transpose(tf.concat(sketches_small_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("original_img_2_l", tf.transpose(tf.concat(sketches_large_list, axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_s1", tf.transpose(tf.concat([t[-1] for t in image_gens_s1], axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_s2", tf.transpose(tf.concat([t[-1] for t in image_gens_s2], axis=0), (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_s1_b",
                             tf.zeros_like(tf.transpose(tf.concat([t[-1] for t in image_gens_s1], axis=0), (0, 2, 3, 1))))
        if distance_map:
            # tf.add_to_collection("dist_map_1_to_2",
            #                      tf.transpose(dist_map_to_image(tf.concat(image_gens_1t2, axis=0)), (0, 2, 3, 1)))
            tf.add_to_collection("dist_map_img_2",
                                 tf.transpose(dist_map_to_image(tf.concat(sketches_large_list, axis=0)), (0, 2, 3, 1)))
            # tf.add_to_collection("dist_map_212",
            #                      tf.transpose(dist_map_to_image(tf.concat(image_gens_212, axis=0)), (0, 2, 3, 1)))

    # Add loss to collections
    tf.add_to_collection("loss_g", loss_g_s2)
    tf.add_to_collection("loss_d", loss_d_s2)

    # Variable Collections
    var_collections = {
        'generator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s1'),
        'discriminator_s1': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s1'),
        'generator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_s2'),
        'discriminator_s2': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_s2'),
        'generator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
        'discriminator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
    }

    ############# Reuse Variables for next tower (?) #############
    tf.get_variable_scope().reuse_variables()
    ############# Reuse Variables for next tower (?) #############

    # # Gather summaries from last tower
    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

    # Calculate Gradient
    grad_g_s2, grad_d_s2 = compute_gradients((loss_g_s2, loss_d_s2),
                                             (optim_g, optim_d),
                                             var_lists=(var_collections['generator_s2'],
                                                        var_collections['discriminator_s2']))

    # return opt_g, opt_d, opt_su_g, loss_g, loss_d, loss_su_g
    return loss_g_s2, loss_d_s2, grad_g_s2, grad_d_s2, (image_gen_100_s2[0],)


def get_losses(discriminator_s1, discriminator_s2, discriminator_e, vae_sampler,
               batch_size0, num_classes, img_dim, channel, stage,
               data_format, ld, perturb_func,
               # images
               image_small_list, sketches_small_list,
               image_large_list, sketches_large_list,
               image_small_d_list,
               image_gens_s1, image_gens_s1_b, image_gens_s2,
               # latent and labels
               image_data_class_id_list,
               image_labels_s1, image_labels_s1_b, image_labels_s2,
               # critic out
               real_disc_out_s1, fake_disc_outs_s1,
               real_disc_out_s2, fake_disc_outs_s2,
               # logit
               real_logit_s1, fake_logits_s1,
               real_logit_s2, fake_logits_s2,
               ):

    def get_acgan_loss(real_image_logits_out, real_image_label,
                       disc_image_logits_out, condition,
                       num_classes, ld1=1, ld2=0.5, ld_focal=2.):
        # loss_ac_d = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
        loss_ac_d = tf.reduce_mean((1 - tf.reduce_sum(tf.nn.softmax(real_image_logits_out) * tf.squeeze(
            tf.one_hot(real_image_label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1)) ** ld_focal *
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
        # loss_ac_d = tf.Print(loss_ac_d, [
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label),
        #     1 - tf.reduce_sum(tf.nn.softmax(real_image_logits_out) * tf.squeeze(
        #         tf.one_hot(real_image_label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1),
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label) * (
        #         1 - tf.reduce_sum(tf.nn.softmax(real_image_logits_out) * tf.squeeze(
        #         tf.one_hot(real_image_label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1))
        # ], summarize=40)
        loss_ac_d = ld1 * loss_ac_d

        loss_ac_g = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
        # loss_ac_g = tf.reduce_mean((1 - tf.reduce_sum(tf.nn.softmax(disc_image_logits_out) * tf.squeeze(
        #     tf.one_hot(condition, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1)) ** ld_focal *
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
        loss_ac_g = ld2 * loss_ac_g
        return loss_ac_g, loss_ac_d

    def get_acgan_loss_0(real_image_logits_out, real_image_label,
                       disc_image_logits_out, condition,
                       ld1=1, ld2=0.1):
        loss_ac_d = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
        loss_ac_d = ld1 * loss_ac_d

        loss_ac_g = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
        loss_ac_g = ld2 * loss_ac_g
        return loss_ac_g, loss_ac_d

    def get_loss_wgan_global_gp(discriminator, img_dim, channel, data_format,
                                fake_data_out, fake_data_out_, real_data_out,
                                fake_data, real_data,
                                scope=None, ld=5):
        assert scope is not None
        if type(fake_data) is list:
            fake_data = fake_data[-1]
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        assert ndim == 4
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_g = -tf.reduce_mean(fake_data_out_)
        loss_d = tf.reduce_mean(fake_data_out) - tf.reduce_mean(real_data_out)
        tf.add_to_collection("WGAN_loss_d", loss_d)

        # Gradient penalty
        batch_size = int(real_data.get_shape()[0])
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = fake_data - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.add_to_collection("WGAN_loss_gp", gradient_penalty)

        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_wgan_one_side_global_gp(discriminator, img_dim, channel, data_format,
                                         fake_data_out, fake_data_out_, real_data_out,
                                         fake_data, real_data,
                                         scope=None, ld=10):
        assert scope is not None
        if type(fake_data) is list:
            fake_data = fake_data[-1]
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        assert ndim == 4
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_g = -tf.reduce_mean(fake_data_out_)
        loss_d = tf.reduce_mean(fake_data_out) - tf.reduce_mean(real_data_out)
        tf.add_to_collection("WGAN_loss_d", loss_d)

        # Gradient penalty
        batch_size = int(real_data.get_shape()[0])
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = fake_data - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        tf.add_to_collection("WGAN_loss_gp", gradient_penalty)

        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_wgan_global_gp_cond(discriminator, img_dim, channel, data_format,
                                     fake_data_out, fake_data_out_, real_data_out,
                                     fake_data, fake_data_cond, real_data, real_data_cond,
                                     scope=None, ld=10):
        assert scope is not None
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(fake_data.get_shape())
        assert ndim == 4
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_g = -tf.reduce_mean(fake_data_out_)
        loss_d = tf.reduce_mean(fake_data_out) - tf.reduce_mean(real_data_out)
        tf.add_to_collection("WGAN_loss_d", loss_d)

        # Concat conditions
        fake_data_fused = tf.concat([fake_data, fake_data_cond], axis=concat_axis)
        real_data_fused = tf.concat([real_data, real_data_cond], axis=concat_axis)

        # Gradient penalty
        batch_size = int(real_data.get_shape()[0])
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = fake_data_fused - real_data_fused
        interp = real_data_fused + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, None, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.add_to_collection("WGAN_loss_gp", gradient_penalty)

        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp(discriminator, img_dim, channel, data_format,
                                       fake_data_out, fake_data_out_, real_data_out,
                                       fake_data, fake_data_cond, real_data, real_data_cond,
                                       scope=None, ld=10):
        assert scope is not None
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(fake_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 2
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_data_out,
                                                                             labels=tf.zeros_like(fake_data_out)))
        loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_data_out,
                                                                             labels=tf.ones_like(real_data_out)))
        loss_g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_data_out,
                                                                             labels=tf.ones_like(fake_data_out)))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])
        # Concat conditions
        fake_data_fused = tf.concat([fake_data, fake_data_cond], axis=concat_axis)
        real_data_fused = tf.concat([real_data, real_data_cond], axis=concat_axis)

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb_func(real_data_fused) - real_data_fused
        interp = real_data_fused + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, None, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp_multi(discriminator, img_dim, channel, data_format,
                                                  fake_data_out, fake_data_out_, real_data_out,
                                                  fake_data, real_data,
                                                  scope=None, ld=10):
        assert scope is not None
        # assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 4 or ndim_out == 2
        if ndim_out == 4:
            sum_axis = (1, 2, 3)
        else:
            sum_axis = 1
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.zeros_like(fake_data_out)), axis=sum_axis))
        loss_d_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_data_out, labels=tf.ones_like(real_data_out)), axis=sum_axis))
        loss_g_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.ones_like(fake_data_out)), axis=sum_axis))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb_func(real_data) - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp_multi_cond(discriminator, img_dim, channel, data_format,
                                                  fake_data_out, fake_data_out_, real_data_out,
                                                  fake_data, fake_data_cond, real_data, real_data_cond,
                                                  scope=None, ld=10):
        assert scope is not None
        # assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 4 or ndim_out == 2
        if ndim_out == 4:
            sum_axis = (1, 2, 3)
        else:
            sum_axis = 1
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.zeros_like(fake_data_out)), axis=sum_axis))
        loss_d_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_data_out, labels=tf.ones_like(real_data_out)), axis=sum_axis))
        loss_g_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.ones_like(fake_data_out)), axis=sum_axis))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])
        # Concat conditions
        # fake_data_fused = tf.concat([fake_data, fake_data_cond], axis=concat_axis)
        real_data_fused = tf.concat([real_data, real_data_cond], axis=concat_axis)

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb_func(real_data_fused) - real_data_fused
        interp = real_data_fused + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, None, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp_one_side_multi(discriminator, img_dim, channel, data_format,
                                                      fake_data_out, fake_data_out_, real_data_out,
                                                      fake_data, real_data,
                                                      scope=None, ld=10):
        assert scope is not None
        # assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 4 or ndim_out == 2
        if ndim_out == 4:
            sum_axis = (1, 2, 3)
        else:
            sum_axis = 1
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.zeros_like(fake_data_out)), axis=sum_axis))
        loss_d_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_data_out, labels=tf.ones_like(real_data_out)), axis=sum_axis))
        loss_g_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.ones_like(fake_data_out)), axis=sum_axis))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb_func(real_data) - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp_one_side_multi_cond(discriminator, img_dim, channel, data_format,
                                                           fake_data_out, fake_data_out_, real_data_out,
                                                           fake_data, fake_data_cond, real_data, real_data_cond,
                                                           scope=None, ld=10):
        assert scope is not None
        # assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 4 or ndim_out == 2
        if ndim_out == 4:
            sum_axis = (1, 2, 3)
        else:
            sum_axis = 1
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.zeros_like(fake_data_out)), axis=sum_axis))
        loss_d_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_data_out, labels=tf.ones_like(real_data_out)), axis=sum_axis))
        loss_g_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.ones_like(fake_data_out)), axis=sum_axis))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])
        # Concat conditions
        # fake_data_fused = tf.concat([fake_data, fake_data_cond], axis=concat_axis)
        real_data_fused = tf.concat([real_data, real_data_cond], axis=concat_axis)

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb_func(real_data_fused) - real_data_fused
        interp = real_data_fused + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, None, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_fisher_gan(discriminator, img_dim, channel, data_format,
                            fake_data_out, fake_data_out_, real_data_out,
                            fake_data, real_data,
                            scope=None, rho=1e-6):
        _rho = rho
        # Initialize alpha (or in paper called lambda) with zero
        # Throughout training alpha is trained with an independent sgd optimizer
        # We use "alpha" instead of lambda because code we are modeling off of
        # uses "alpha" instead of lambda
        with tf.variable_scope(scope) as sc:
            _alpha = tf.get_variable("fisher_alpha", [], initializer=tf.zeros_initializer)

        # Compared to WGAN, generator cost remains the same in fisher GAN
        loss_g = -tf.reduce_mean(fake_data_out_)

        # Calculate Lipchitz Constraint
        # E_P and E_Q refer to Expectation over real and fake.
        E_Q_f = tf.reduce_mean(fake_data_out)
        E_P_f = tf.reduce_mean(real_data_out)
        E_Q_f2 = tf.reduce_mean(fake_data_out ** 2)
        E_P_f2 = tf.reduce_mean(real_data_out ** 2)

        constraint = (1 - (0.5 * E_P_f2 + 0.5 * E_Q_f2))

        # See Equation (9) in Fisher GAN paper
        # In the original implementation, they use a backward computation with mone (minus one)
        # To implement this in tensorflow, we simply multiply the objective
        # cost function by minus one.
        loss_d = -1. * (E_P_f - E_Q_f + _alpha * constraint - _rho / 2. * constraint ** 2.)

        # Find gradient of alpha with respect to negative disc_cost
        _alpha_optimizer = tf.train.GradientDescentOptimizer(_rho)
        alpha_optimizer_op = _alpha_optimizer.minimize(-loss_d, var_list=[_alpha])
        tf.add_to_collection("fisher_alpha", _alpha)
        # loss_d = tf.Print(loss_d, [_alpha, loss_d])

        with tf.control_dependencies([alpha_optimizer_op]):
            loss_d = tf.identity(loss_d)

        return loss_g, loss_d

    def get_loss_fisher_gan_cond(discriminator, img_dim, channel, data_format,
                                 fake_data_out, fake_data_out_, real_data_out,
                                 fake_data, fake_data_cond, real_data, real_data_cond,
                                 scope=None, rho=1e-6):
        _rho = rho
        # Initialize alpha (or in paper called lambda) with zero
        # Throughout training alpha is trained with an independent sgd optimizer
        # We use "alpha" instead of lambda because code we are modeling off of
        # uses "alpha" instead of lambda
        with tf.variable_scope(scope) as sc:
            _alpha = tf.get_variable("fisher_alpha", [], initializer=tf.zeros_initializer)

        # Compared to WGAN, generator cost remains the same in fisher GAN
        loss_g = -tf.reduce_mean(fake_data_out_)

        # Calculate Lipchitz Constraint
        # E_P and E_Q refer to Expectation over real and fake.
        E_Q_f = tf.reduce_mean(fake_data_out)
        E_P_f = tf.reduce_mean(real_data_out)
        E_Q_f2 = tf.reduce_mean(fake_data_out ** 2)
        E_P_f2 = tf.reduce_mean(real_data_out ** 2)

        constraint = (1 - (0.5 * E_P_f2 + 0.5 * E_Q_f2))

        # See Equation (9) in Fisher GAN paper
        # In the original implementation, they use a backward computation with mone (minus one)
        # To implement this in tensorflow, we simply multiply the objective
        # cost function by minus one.
        loss_d = -1. * (E_P_f - E_Q_f + _alpha * constraint - _rho / 2. * constraint ** 2.)

        # Find gradient of alpha with respect to negative disc_cost
        _alpha_optimizer = tf.train.GradientDescentOptimizer(_rho)
        alpha_optimizer_op = _alpha_optimizer.minimize(-loss_d, var_list=[_alpha])
        tf.add_to_collection("fisher_alpha", _alpha)
        # loss_d = tf.Print(loss_d, [_alpha, loss_d])

        with tf.control_dependencies([alpha_optimizer_op]):
            loss_d = tf.identity(loss_d)

        return loss_g, loss_d

    def get_loss_cramer_gan_global_gp(discriminator, img_dim, channel, data_format,
                                      fake_data_out, fake_data_out_, real_image_out,
                                      fake_data, real_data,
                                      scope=None, ld=10):
        def l2(tensor):
            # Assume first dimension is batch
            input_shape = tensor.get_shape().as_list()
            batch_size = input_shape[0]
            ndim = len(input_shape)
            tensor = tf.reshape(tensor, shape=(batch_size, -1))
            reduce_axis = 1
            # reduce_axis = list(range(1, ndim))
            # if len(reduce_axis) == 1:
            #     reduce_axis = reduce_axis[0]
            # return tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=reduce_axis))
            return tf.norm(tensor=tensor, ord='euclidean', axis=reduce_axis)

        assert scope is not None
        if type(fake_data) is list:
            fake_data = fake_data[-1]
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        assert ndim == 4

        def critic(x):
            return l2(x - fake_data_out_) - l2(x)

        # loss_g = l2(real_image_out - fake_image_out_111) + l2(real_image_out - fake_image_prime_out) - l2(
        #     fake_image_out_111 - fake_image_prime_out)
        # loss_g = tf.Print(loss_g, [loss_g, l2(real_image_out - fake_image_out), l2(real_image_out - fake_image_prime_out),
        #                            l2(fake_image_out - fake_image_prime_out)])

        loss_g = tf.reduce_mean(critic(real_image_out) - critic(fake_data_out))
        loss_d = -loss_g
        # loss_d = tf.Print(loss_d, [loss_d, critic(fake_image_out), critic(real_image_out)])

        # Gradient penalty
        batch_size = int(real_data.get_shape()[0])
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1., dtype=tf.float32)
        diff = fake_data - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(critic(discriminator(interp, num_classes=num_classes, reuse=True,
                                                      data_format=data_format, scope_name=scope)[0]),
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def build_inception(inputs, reuse=True, scope='InceptionV4'):
        is_training = False
        arg_scope = inception_v4_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                    logits, end_points = inception_v4_base(inputs, final_endpoint='Mixed_5b', scope=scope)
            # return inception_v4(input, num_classes, is_training=False)
        return [end_points['Conv2d_2a_3x3'], end_points['Mixed_4a'], end_points['Mixed_5b']]
        # return [end_points['Mixed_4a'], end_points['Mixed_5b'], end_points['Mixed_6b']]
        # return [end_points['Conv2d_2a_3x3'], end_points['Mixed_4a']]

    def build_vgg(inputs, reuse=True, scope='vgg_16', num=0):

        def get_endpoint(end_points, name):
            for key in end_points.keys():
                if name in key:
                    return end_points[key]

        is_training = False
        arg_scope = vgg_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, end_points, my_end_points = vgg_16(inputs, is_training=is_training,
                                                       reuse=reuse, scope=scope, num=num)
        # return [end_points['conv1_2'], end_points['conv2_2'], end_points['conv3_2'], end_points['conv4_2'], ]
        return [get_endpoint(end_points, 'conv1_2'), get_endpoint(end_points, 'conv2_2'),
                get_endpoint(end_points, 'conv3_2'), get_endpoint(end_points, 'conv4_2'), ]
        # return [logits,end_points['vgg_16/fc8'],logits,logits]

    def get_perceptual_loss(image1, image2, data_format, type="Inception", reuse=True):
        assert data_format == 'NCHW'
        # global vgg_assign_fn
        # vgg_lys = []
        # vgg_assign_fn = vgg.get_assign_func(batch_size0)

        image1 = tf.transpose(image1, (0, 2, 3, 1))
        image2 = tf.transpose(image2, (0, 2, 3, 1))

        if type == "Inception":
            # Normalize to 0-1
            image1 = (image1 + 1) / 2.
            image2 = (image2 + 1) / 2.

            dim = 299

            # Resize to 299, 299
            image1 = tf.image.resize_bilinear(image1, [dim, dim])
            image2 = tf.image.resize_bilinear(image2, [dim, dim])

            image1_lys = build_inception(image1, reuse=reuse)
            image2_lys = build_inception(image2)
        elif type == "vgg":
            image_size = image1.get_shape().as_list()

            dim = 224

            _R_MEAN = tf.constant(123.68, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)
            _G_MEAN = tf.constant(116.78, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)
            _B_MEAN = tf.constant(103.94, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)

            _MEAN = tf.concat([_R_MEAN, _G_MEAN, _B_MEAN], axis=3)

            # Normalize to 0-255
            image1 = (image1 + 1) * 255. / 2.
            image2 = (image2 + 1) * 255. / 2.

            # Resize to 299, 299
            image1 = tf.image.resize_bilinear(image1, [dim, dim])
            image2 = tf.image.resize_bilinear(image2, [dim, dim])

            # Substract mean
            image1 -= _MEAN
            image2 -= _MEAN

            # image1_lys = build_inception(image1, reuse=reuse)
            # image2_lys = build_inception(image2)

            image1_lys = build_vgg(image1, reuse=reuse, num=0)
            image2_lys = build_vgg(image2, num=1)
        else:
            raise ValueError("Network type unknown.")

        tf.add_to_collection("inception_layer_1_1", image1_lys[0])
        tf.add_to_collection("inception_layer_1_2", image1_lys[1])
        tf.add_to_collection("inception_layer_1_3", image1_lys[2])
        # tf.add_to_collection("inception_layer_1_4", image1_lys[3])
        # tf.add_to_collection("inception_layer_1_5", image1_lys[4])

        tf.add_to_collection("inception_layer_2_1", image2_lys[0])
        tf.add_to_collection("inception_layer_2_2", image2_lys[1])
        tf.add_to_collection("inception_layer_2_3", image2_lys[2])
        # tf.add_to_collection("inception_layer_2_4", image2_lys[3])
        # tf.add_to_collection("inception_layer_2_5", image2_lys[4])

        loss_perceptual = 0.
        # coeffs = [1., 0.5, 0.25, 0.05]
        for i in range(len(image2_lys)):
            loss_perceptual += tf.reduce_mean(tf.abs(image2_lys[i] - image1_lys[i]))    # L1
            # loss_perceptual += coeffs[i] * tf.sqrt(tf.reduce_sum(tf.square(image2_lys[i] - image1_lys[i]), axis=[1, 2, 3]))    # L2
            # loss_perceptual = coeffs[i] * models.vae_loss_reconstruct(image2_lys[i], image1_lys[i])       # log-likelihood
        return loss_perceptual

    assert stage == 1 or stage == 2
    get_gan_loss = get_loss_original_gan_local_gp_one_side_multi
    # get_gan_loss_cond = get_loss_original_gan_local_gp_multi_cond

    # upsample and downsample
    if stage == 2:
        s1_dims = image_gens_s1[0][-1].get_shape().as_list()
        s2_dims = image_gens_s2[0][-1].get_shape().as_list()
        assert len(image_gens_s1) == len(image_gens_s2)
        image_gens_s1_large = []
        image_gens_s2_small = []
        if s1_dims[2] < s2_dims[2]:
            multi = int(np.log2((s2_dims[2] / s1_dims[2])))
            assert multi == np.log2(s2_dims[2] / s1_dims[2])

            for i in range(len(image_gens_s1)):
                image_gen_s1_large = image_gens_s1[i][-1]
                image_gen_s2_small = image_gens_s2[i][-1]
                for j in range(multi):
                    image_gen_s1_large = upsample(image_gen_s1_large, data_format)
                    image_gen_s2_small = mean_pool(image_gen_s2_small, data_format)
                image_gens_s1_large.append(image_gen_s1_large)
                image_gens_s2_small.append(image_gen_s2_small)
        elif s1_dims[2] == s2_dims[2]:
            for i in range(len(image_gens_s1)):
                image_gens_s1_large.append(image_gens_s1[i][-1])
                image_gens_s2_small.append(image_gens_s2[i][-1])
            print("Stage1 and Stage2 dimensionality is equal.")
        else:
            raise ValueError

    if stage == 1:
        # GAN Loss, stage 1
        loss_g_s1, loss_d_s1 = get_gan_loss(discriminator_s1, img_dim, channel, data_format,
                                            fake_disc_outs_s1[0],
                                            fake_disc_outs_s1[0],
                                            real_disc_out_s1,
                                            image_gens_s1[0],
                                            image_small_d_list[1],
                                            scope='discriminator_s1')
        # loss_g_s1, loss_d_s1 = get_gan_loss_cond(discriminator_s1, img_dim, channel, data_format,
        #                                          fake_disc_outs_s1[0],
        #                                          fake_disc_outs_s1[0],
        #                                          real_disc_out_s1,
        #                                          image_gens_s1[0], sketches_small_list[0],
        #                                          image_small_list[1], sketches_small_list[1],
        #                                          scope='discriminator_s1')
        tf.add_to_collection("DRAGAN_loss_g_s1", loss_g_s1)
        tf.add_to_collection("DRAGAN_loss_d_s1", loss_d_s1)
        # GAN Loss
        loss_d_gan = loss_d_s1
        loss_g_gan = loss_g_s1
        tf.add_to_collection("DRAGAN_loss_g", loss_g_gan)
        tf.add_to_collection("DRAGAN_loss_d", loss_d_gan)

    elif stage == 2:
        # GAN Loss, stage 2
        loss_g_s2, loss_d_s2 = get_gan_loss(discriminator_s2, img_dim, channel, data_format,
                                            fake_disc_outs_s2[0],
                                            fake_disc_outs_s2[0],
                                            real_disc_out_s2,
                                            image_gens_s2[0],
                                            image_large_list[1],
                                            scope='discriminator_s2')
        # loss_g_s2, loss_d_s2 = get_gan_loss_cond(discriminator_s2, img_dim, channel, data_format,
        #                                          fake_disc_outs_s2[0],
        #                                          fake_disc_outs_s2[0],
        #                                          real_disc_out_s2,
        #                                          image_gens_s2[0], sketches_large_list[0],
        #                                          image_large_list[1], sketches_large_list[1],
        #                                          scope='discriminator_s2')
        tf.add_to_collection("DRAGAN_loss_g_s2", loss_g_s2)
        tf.add_to_collection("DRAGAN_loss_d_s2", loss_d_s2)
        # GAN Loss
        loss_d_gan = loss_d_s2
        loss_g_gan = loss_g_s2
        tf.add_to_collection("DRAGAN_loss_g", loss_g_gan)
        tf.add_to_collection("DRAGAN_loss_d", loss_d_gan)

    # ACGAN loss
    if stage == 1:
        loss_ac_g_s1, loss_ac_d_s1 = get_acgan_loss(real_logit_s1, image_data_class_id_list[1],
                                                    fake_logits_s1[0], image_labels_s1[0],
                                                    num_classes=num_classes)
        tf.add_to_collection("ACGAN_loss_g_s1", loss_ac_g_s1)
        tf.add_to_collection("ACGAN_loss_d_s1", loss_ac_d_s1)
        loss_ac_g = loss_ac_g_s1
        loss_ac_d = loss_ac_d_s1
        tf.add_to_collection("ACGAN_loss_g", loss_ac_g)
        tf.add_to_collection("ACGAN_loss_d", loss_ac_d)
    elif stage == 2:
        loss_ac_g_s2, loss_ac_d_s2 = get_acgan_loss(real_logit_s2, image_data_class_id_list[1],
                                                    fake_logits_s2[0], image_labels_s2[0],
                                                    num_classes=num_classes)
        tf.add_to_collection("ACGAN_loss_g_s2", loss_ac_g_s2)
        tf.add_to_collection("ACGAN_loss_d_s2", loss_ac_d_s2)
        loss_ac_g = loss_ac_g_s2
        loss_ac_d = loss_ac_d_s2
        tf.add_to_collection("ACGAN_loss_g", loss_ac_g)
        tf.add_to_collection("ACGAN_loss_d", loss_ac_d)

    loss_g_gan += loss_ac_g
    loss_d_gan += loss_ac_d

    if stage == 1:
        # Supervised loss
        loss_gt_s1 = 0.
        for i in range(1):
            loss_gt_s1 += tf.losses.absolute_difference(image_small_list[i], image_gens_s1[i][-1])  # L1
            loss_gt_s1 += 0.3 * get_perceptual_loss(image_small_list[i], image_gens_s1[i][-1],
                                                    data_format=data_format, type="Inception", reuse=False)  # perceptual
        loss_gt = loss_gt_s1
        tf.add_to_collection("direct_loss_s1", loss_gt_s1)
        tf.add_to_collection("direct_loss", loss_gt)

        # Diverse loss
        loss_dv = 0.
        for i in range(1):
            loss_dv -= tf.losses.absolute_difference(image_gens_s1[i][-1], image_gens_s1_b[i][-1])  # L1
        tf.add_to_collection("diverse_loss", loss_dv)
    elif stage == 2:
        # # Supervised loss
        # loss_gt_s2 = 0.
        # for i in range(1):
        #     loss_gt_s2 += 0.4 * get_perceptual_loss(image_large_list[i], image_gens_s2[i],
        #                                             data_format=data_format, type="Inception", reuse=False)  # perceptual
        # loss_gt = loss_gt_s2
        # tf.add_to_collection("direct_loss_s2", loss_gt_s2)
        # tf.add_to_collection("direct_loss", loss_gt)

        # Regularization loss
        loss_r = 0.
        for i in range(1):
            loss_r += tf.losses.absolute_difference(image_gens_s2_small[i], image_gens_s1[i][-1])  # L1
        tf.add_to_collection("regularization_loss", loss_r)

    coeff_gt = 10
    coeff_dv = 1
    coeff_r = 1

    if stage == 1:
        loss_g_s1 = loss_g_s1 + loss_ac_g_s1 + coeff_gt * loss_gt_s1 + coeff_dv * loss_dv
        loss_d_s1 = loss_d_s1 + loss_ac_d_s1
        tf.add_to_collection("total_loss_g_s1", loss_g_s1)
        tf.add_to_collection("total_loss_d_s1", loss_d_s1)
        loss_g = loss_g_s1
        loss_d = loss_d_s1
    elif stage == 2:
        loss_g_s2 = loss_g_s2 + loss_ac_g_s2 + coeff_r * loss_r
        loss_d_s2 = loss_d_s2 + loss_ac_d_s2
        tf.add_to_collection("total_loss_g_s2", loss_g_s2)
        tf.add_to_collection("total_loss_d_s2", loss_d_s2)
        loss_g = loss_g_s2
        loss_d = loss_d_s2
    # loss_g_s2 = loss_g_s2 + loss_ac_g_s2 + coeff_gt * loss_gt_s2 + coeff_r * loss_r
    # loss_d_s2 = loss_d_s2 + loss_ac_d_s2
    # loss_g = loss_g_gan + coeff_gt * loss_gt + coeff_dv * loss_dv + coeff_r * loss_r
    # loss_d = loss_d_gan

    tf.add_to_collection("total_loss_g", loss_g)
    tf.add_to_collection("total_loss_d", loss_d)

    return loss_g, loss_d


def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name.lower() == 'RMSProp'.lower():
        # decay=0.9, momentum=0.0, epsilon=1e-10
        return functools.partial(tf.train.RMSPropOptimizer, decay=0.9, momentum=0.0, epsilon=1e-10)
    elif optimizer_name.lower() == 'Adam'.lower():
        # beta1=0.5, beta2=0.9, epsilon=1e-8
        # return functools.partial(tf.train.AdamOptimizer, beta1=0., beta2=0.9)
        return functools.partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9)
    elif optimizer_name.lower() == 'AdaDelta'.lower():
        return tf.train.AdadeltaOptimizer
    elif optimizer_name.lower() == 'AdaGrad'.lower():
        return tf.train.AdagradOptimizer


def optimize(gradients, optim, global_step, summaries, global_norm=None, global_norm_clipped=None, appendix=''):
    """Modified from sugartensor"""

    # Add Summary
    if summaries is None:
        summaries = ["loss", "learning_rate"]
    if "gradient_norm" in summaries:
        if global_norm is None:
            tf.summary.scalar("global_norm/gradient_norm" + appendix,
                              clip_ops.global_norm(list(zip(*gradients))[0]))
        else:
            tf.summary.scalar("global_norm/gradient_norm" + appendix,
                              global_norm)
        if global_norm_clipped is not None:
            tf.summary.scalar("global_norm/gradient_norm_clipped" + appendix,
                              global_norm_clipped)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if grad_values is not None:
            var_name = variable.name.replace(":", "_")
            if "gradients" in summaries:
                tf.summary.histogram("gradients/%s" % var_name, grad_values)
            if "gradient_norm" in summaries:
                tf.summary.scalar("gradient_norm/%s" % var_name,
                                  clip_ops.global_norm([grad_values]))

    # Gradient Update OP
    return optim.apply_gradients(gradients, global_step=global_step)
