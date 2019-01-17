import argparse
import importlib
import os
import sys
import shutil
import json
import tensorflow as tf
from time import gmtime, strftime

src_dir = './src'


def launch_training(**kwargs):

    # Deal with file and paths
    appendix = kwargs["resume_from"]
    if appendix is None or appendix == '':
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_dir = './log_wgan_' + cur_time
        ckpt_dir = './ckpt_wgan_' + cur_time
        if not os.path.isdir(log_dir) and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.isdir(ckpt_dir) and not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # copy current script in src folder to log dir for record
        if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
            print("src folder does not exist.")
            return
        else:
            for file in os.listdir(src_dir):
                if file.endswith(".py"):
                    shutil.copy(os.path.join(src_dir, file), log_dir)

        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        appendix = cur_time

        # Save parameters
        with open(os.path.join(log_dir, 'param_%d.json' % 0), 'w') as fp:
            json.dump(kwargs, fp, indent=4)

        sys.path.append(src_dir)
        entry_point_module = kwargs['entry_point']

        print("Launching new train: %s" % cur_time)
    else:
        # curr_dir = os.path.split(inspect.getfile(inspect.currentframe()))
        # this_dir = os.path.split(curr_dir[0])[1]
        # appendix = this_dir.split('_')[2]
        if len(appendix.split('-')) != 6:
            print("Invalid resume folder")
            return

        log_dir = './log_wgan_' + appendix
        ckpt_dir = './ckpt_wgan_' + appendix

        # Get last parameters (recover entry point module name)
        json_files = [f for f in os.listdir(log_dir) if
                      os.path.isfile(os.path.join(log_dir, f)) and os.path.splitext(f)[1] == '.json']
        iter_starts = max([int(os.path.splitext(filename)[0].split('_')[1]) for filename in json_files])
        with open(os.path.join(log_dir, 'param_%d.json' % iter_starts), 'r') as fp:
            params = json.load(fp)
        entry_point_module = params['entry_point']

        # Recover some parameters
        kwargs["batch_size"] = params["batch_size"]
        kwargs["img_dim"] = params["img_dim"]
        kwargs["num_classes"] = params["num_classes"]
        kwargs["noise_dim"] = params["noise_dim"]
        kwargs["max_iter_step"] = params["max_iter_step"]
        kwargs["disc_iterations"] = params["disc_iterations"]
        kwargs["lambda"] = params["lambda"]
        kwargs["optimizer"] = params["optimizer"]
        kwargs["lr_G"] = params["lr_G"]
        kwargs["lr_D"] = params["lr_D"]
        kwargs["data_format"] = params["data_format"]
        kwargs["distance_map"] = params["distance_map"]
        kwargs["small_img"] = params["small_img"]
        # kwargs["extra_info"] = params["extra_info"]

        stage_1_log_dir = os.path.join(log_dir, "stage1")
        stage_1_ckpt_dir = os.path.join(ckpt_dir, "stage1")
        if not os.path.exists(stage_1_log_dir) or not os.path.exists(stage_1_ckpt_dir):
            raise RuntimeError
        stage_2_log_dir = os.path.join(log_dir, "stage2")
        stage_2_ckpt_dir = os.path.join(ckpt_dir, "stage2")

        if not os.path.exists(stage_2_ckpt_dir) or tf.train.latest_checkpoint(stage_2_ckpt_dir) is None:
            stage = 1
        else:
            stage = 2

        sys.path.append(log_dir)

        # Get latest checkpoint filename
        if stage == 1:
            ckpt_file = tf.train.latest_checkpoint(stage_1_ckpt_dir)
        elif stage == 2:
            ckpt_file = tf.train.latest_checkpoint(stage_2_ckpt_dir)
        if ckpt_file is None:
            raise RuntimeError
        else:
            iter_from = int(os.path.split(ckpt_file)[1].split('-')[1]) + 1
            if stage == 2:
                iter_from += int(kwargs["max_iter_step"] / 2)
        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        kwargs['iter_from'] = iter_from
        # kwargs['restart_from'] = appendix

        # Save new set of parameters
        with open(os.path.join(log_dir, 'param_%d.json' % iter_from), 'w') as fp:
            kwargs['entry_point'] = entry_point_module
            json.dump(kwargs, fp, indent=4)

        print("Launching train from checkpoint: %s" % appendix)

    # Launch train
    train_module = importlib.import_module(entry_point_module)
    # from train_paired_aug_multi_gpu import train
    status = train_module.train(**kwargs)

    return status, appendix


def launch_test(**kwargs):
    # Deal with file and paths
    appendix = kwargs["resume_from"]
    if appendix is None or appendix == '' or len(appendix.split('-')) != 6:
        print("Invalid resume folder")
        return

    log_dir = './log_wgan_' + appendix
    ckpt_dir = './ckpt_wgan_' + appendix

    sys.path.append(log_dir)

    # Get latest checkpoint filename
    kwargs['log_dir'] = log_dir
    kwargs['ckpt_dir'] = ckpt_dir

    # Get last parameters (recover entry point module name)
    # Assuming last json file
    json_files = [f for f in os.listdir(log_dir) if
                  os.path.isfile(os.path.join(log_dir, f)) and os.path.splitext(f)[1] == '.json']
    iter_starts = max([int(os.path.splitext(filename)[0].split('_')[1]) for filename in json_files])
    with open(os.path.join(log_dir, 'param_%d.json' % iter_starts), 'r') as fp:
        params = json.load(fp)
    entry_point_module = params['entry_point']

    # Recover some parameters
    kwargs["batch_size"] = params["batch_size"]
    kwargs["img_dim"] = params["img_dim"]
    kwargs["num_classes"] = params["num_classes"]
    kwargs["noise_dim"] = params["noise_dim"]
    kwargs["max_iter_step"] = params["max_iter_step"]
    kwargs["disc_iterations"] = params["disc_iterations"]
    kwargs["lambda"] = params["lambda"]
    kwargs["optimizer"] = params["optimizer"]
    kwargs["lr_G"] = params["lr_G"]
    kwargs["lr_D"] = params["lr_D"]
    kwargs["data_format"] = params["data_format"]
    kwargs["distance_map"] = params["distance_map"]
    kwargs["small_img"] = params["small_img"]
    # kwargs["extra_info"] = params["extra_info"]

    print("Launching test from checkpoint: %s" % appendix)

    # Launch test
    train_module = importlib.import_module(entry_point_module)
    train_module.test(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or Test model')
    parser.add_argument('--mode', type=str, default="test", help="train or test")
    parser.add_argument('--resume_from', type=str, default='2017-10-28-06-01-04', help="Whether resume last checkpoint from a past run")
    parser.add_argument('--entry_point', type=str, default='train_hed_stack_separate', help="Whether resume last checkpoint from a past run")
    # parser.add_argument('--dset1', type=str, default="kitti", help="mnist or svhn or cifar10")
    # parser.add_argument('--dset2', type=str, default="pfd", help="mnist or svhn or cifar10")
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per gpu')
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height")
    parser.add_argument('--num_classes', default=50, type=int, help="Number of classes")
    parser.add_argument('--noise_dim', default=128, type=int, help="noise sampler dimension")
    parser.add_argument('--max_iter_step', default=600000, type=int, help="Max number of iterations")
    parser.add_argument('--weight_decay_rate', default=4e-5, type=float, help="Weight decay rate for ResNet blocks")
    parser.add_argument('--deconv_weight_decay_rate', default=1e-5, type=float, help="Weight decay rate for ResNet deconv blocks")
    parser.add_argument('--disc_iterations', default=1, type=int, help="Number of discriminator iterations")
    parser.add_argument('--ld', default=10, type=float, help="Gradient penalty lambda hyperparameter")
    # parser.add_argument('--clamp_lower', default=-0.01, type=float, help="Clamp weights below this value")
    # parser.add_argument('--clamp_upper', default=0.01, type=float, help="Clamp weights above this value")
    parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer for the graph")
    parser.add_argument('--lr_G', type=float, default=1e-4, help="learning rate for the generator")
    parser.add_argument('--lr_D', type=float, default=2e-4, help="learning rate for the discriminator")
    # parser.add_argument('--device', default='/gpu:0', type=str, help="Default device to run graph")
    parser.add_argument('--num_gpu', default=1, type=int, help="Number of GPUs to use")
    parser.add_argument('--data_format', default='NCHW', type=str, help="Default data format")
    parser.add_argument('--distance_map', default=1, type=int, help="Whether using distance maps for sketches")
    parser.add_argument('--small_img', default=0, type=int, help="Whether using 64x64 instead of 256x256")
    parser.add_argument('--test_image_folder', default='./test_images', type=str, help="Path to folder holding test images")
    parser.add_argument('--stage', default=1, type=int, help="Direction of transformation tested")
    parser.add_argument('--extra_info', default="", type=str, help="Extra information saved for record")

    args = parser.parse_args()

    # assert args.dset1 in ["mnist", "mnist3", "mnistm", "svhn", "cifar10"]
    # assert args.dset2 in ["mnist", "mnist3", "mnistm", "svhn", "cifar10"]
    assert args.optimizer in ["RMSprop", "Adam0", "Adam", "AdaDelta", "AdaGrad"], "Unsupported optimizer"

    # Set default params
    d_params = {"resume_from": args.resume_from,
                "entry_point": args.entry_point,
                # "dset1": args.dset1,
                # "dset2": args.dset2,
                "batch_size": args.batch_size,
                "img_dim": args.img_dim,
                "num_classes": args.num_classes,
                "noise_dim": args.noise_dim,
                "max_iter_step": args.max_iter_step,
                "weight_decay_rate": args.weight_decay_rate,
                "deconv_weight_decay_rate": args.deconv_weight_decay_rate,
                "disc_iterations": args.disc_iterations,
                "lambda": args.ld,
                "optimizer": args.optimizer,
                "lr_G": args.lr_G,
                "lr_D": args.lr_D,
                # "device": args.device,
                "num_gpu": args.num_gpu,
                "data_format": args.data_format,
                "distance_map": args.distance_map,
                "small_img": args.small_img,
                "test_image_folder": args.test_image_folder,
                "stage": args.stage,
                "extra_info": args.extra_info,
                }

    if args.mode == 'train':
        # Launch training
        status, appendix = launch_training(**d_params)
        while status == -1:  # NaN during training
            print("Training ended with status -1. Restarting..")
            d_params["resume_from"] = appendix
            status = launch_training(**d_params)
    elif args.mode == 'test':
        launch_test(**d_params)
