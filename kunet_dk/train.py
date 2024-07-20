import argparse
import os

from tensorflow.python.keras.callbacks import EarlyStopping

from kunet_dk.improve_ref_masks_callback import ImproveRefMasksCallback
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
    training_eval_step_ratio = config['kunet_dk']['training_eval_step_ratio']
    mask_weaken_modifier = config['kunet_dk']['mask_weaken_modifier']
    mask_weaken_modifier_decay = config['kunet_dk']['mask_weaken_modifier_decay']
    patience = config['kunet_dk']['patience']
    patience_increase = config['kunet_dk']['patience_increase']
    global_patience = config['kunet_dk']['global_patience']

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
    print(f'training_eval_step_ratio: {training_eval_step_ratio}')
    print(f'mask_weaken_modifier: {mask_weaken_modifier}')
    print(f'mask_weaken_modifier_decay: {mask_weaken_modifier_decay}')
    print(f'patience: {patience}')
    print(f'patience_increase: {patience_increase}')
    print(f'global_patience: {global_patience}')

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

    net = Unetlike([*net_input_size, 6],
                   get_experiment_model_name(experiment_name, fold_no), experiment_dir)

    train_gen = DataGenerator(train_imgs, train_masks, batch_size, net_input_size,
                              training=True,
                              max_queue_size=50,
                              workers=7,
                              use_multiprocessing=False)
    val_gen = DataGenerator(val_imgs, val_masks, batch_size, net_input_size,
                            max_queue_size=50,
                            workers=7,
                            use_multiprocessing=False)

    improve_masks_callback = ImproveRefMasksCallback(
        net, training_eval_step_ratio, net_input_size, train_imgs, train_masks, val_imgs, val_masks,
        mask_weaken_modifier, mask_weaken_modifier_decay, patience, patience_increase)

    early_stopping = EarlyStopping(patience=global_patience)

    history = net.fit(train_gen, val_gen,
                      epochs=epochs,
                      initial_epoch=0,
                      training_verbosity=2,
                      additional_callbacks=[improve_masks_callback, early_stopping])

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
