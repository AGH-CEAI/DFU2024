import argparse
import os

from kunet_dk.unetlike import Unetlike
from kunet_dk.data_generator import DataGenerator

from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, plot_and_save_fig, \
    get_experiment_model_name, get_experiment_dir


def main(config):
    fold_no = config['kunet_dk']['fold_no']
    folds_count = config['kunet_dk']['folds_count']
    imgs_dirs = config['kunet_dk']['imgs_dirs']
    masks_dirs = config['kunet_dk']['masks_dirs']
    batch_size = config['kunet_dk']['batch_size']
    extract_test_set = config['kunet_dk']['extract_test_set']
    epochs = config['kunet_dk']['epochs']
    experiment_name = config['kunet_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_root_dir = config['experiment_artifacts_root_dir']
    net_input_size = config['kunet_dk']['net_input_size']

    experiment_dir = get_experiment_dir(experiment_artifacts_root_dir, experiment_name, experiment_type)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f'fold_no: {fold_no}')
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dirs}')
    print(f'masks_dir: {masks_dirs}')
    print(f'batch_size: {batch_size}')
    print(f'epochs: {epochs}')
    print(f'experiment_name: {experiment_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_root_dir}')
    print(f'net_input_size: {net_input_size}')

    train_set, val_set, test_set = [], [], []
    for db_idx, (imgs_dir, masks_dir) in enumerate(zip(imgs_dirs, masks_dirs)):
        imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

        training, valid, test = get_foldwise_split(db_idx, fold_no, folds_count, imgs_masks_pairs,
                                                   read_from_file=False, save_split_to_file=True,
                                                   with_test_set=extract_test_set[db_idx])
        train_set += training
        val_set += valid
        test_set += test

    train_imgs, train_masks = read_imgs_with_masks(train_set)
    val_imgs, val_masks = read_imgs_with_masks(val_set)
    #test_imgs, test_masks = read_imgs_with_masks(test_set)

    train_gen = DataGenerator(train_imgs, train_masks, batch_size, net_input_size, training=True,
                              max_queue_size=500,
                              workers=7,
                              use_multiprocessing=False)
    val_gen = DataGenerator(val_imgs, val_masks, batch_size, net_input_size,
                            max_queue_size=500,
                            workers=7,
                            use_multiprocessing=False)

    net = Unetlike([*net_input_size, 6],
                   get_experiment_model_name(experiment_name, fold_no), experiment_dir)

    history = net.fit(train_gen, val_gen,
                      epochs=epochs,
                      initial_epoch=0,
                      training_verbosity=1)

    plot_and_save_fig([history.history['loss'], history.history['val_loss']],
                      ['training', 'validation'],
                      'epoch', 'loss',
                      os.path.join(experiment_dir, f'fold_{fold_no}_loss_{experiment_name}'))

    plot_and_save_fig([history.history['accuracy'], history.history['val_accuracy']],
                      ['training', 'validation'],
                      'epoch', 'accuracy',
                      os.path.join(experiment_dir, f'fold_{fold_no}_accuracy_{experiment_name}'))



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('fold_no', type=int, help='fold number to train.')
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')
    arg_parser.add_argument('batch_size', type=int, help='size of batch during training')
    arg_parser.add_argument('epochs', type=int, help='number of epochs')
    arg_parser.add_argument('--experiment_name', type=str, default='segm',
                            help='needed to define model name, it will be like experiment_name_fold_no.h5')
    args = arg_parser.parse_args()
    main(args.fold_no,
         args.folds_count,
         args.imgs_dir,
         args.masks_dir,
         args.batch_size,
         args.epochs,
         args.experiment_name)
