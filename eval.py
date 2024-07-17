import argparse
import csv
import json

import numpy as np

from utils import load_files, read_images, read_config


def main(config):
    ref_masks_dir = config['ref_masks_dir']
    pred_masks_dir = config['pred_masks_dir']
    split_file_path = config['split_file_path']
    eval_report_output = config['eval_report_output']

    print(f'ref_masks_dir: {ref_masks_dir}')
    print(f'pred_masks_dir: {pred_masks_dir}')
    print(f'split_file_path: {split_file_path}')
    print(f'eval_report_output: {eval_report_output}')

    if split_file_path is not None:
        with open(split_file_path, 'r') as f:
            content = json.load(f)
            test_set = content['test']
            test_set = [mask_file for image_file, mask_file in test_set]
    else:
        test_set = None

    ref_masks_paths = load_files(ref_masks_dir, test_set=test_set, img_file_pattern="*.png")
    ref_masks_paths = sorted(ref_masks_paths)
    ref_masks = read_images(ref_masks_paths)

    pred_masks_paths = load_files(pred_masks_dir, test_set=test_set, img_file_pattern="*.png")
    pred_masks_paths = sorted(pred_masks_paths)
    pred_masks = read_images(pred_masks_paths)

    header = ['file name', 'dice image-wise']
    report = []
    dice_sum = 0.0
    for i, (ref_mask_key, pred_mask_key) in enumerate(zip(ref_masks, pred_masks)):
        assert ref_mask_key == pred_mask_key

        pred_mask = pred_masks[pred_mask_key]
        pred_mask = pred_mask.astype(np.float32)
        thresh = pred_mask.max()/2
        pred_mask[pred_mask < thresh] = 0.0
        pred_mask[pred_mask >= thresh] = 1.0

        ref_mask = ref_masks[ref_mask_key]
        ref_mask = ref_mask.astype(np.float32)
        thresh = ref_mask.max()/2
        ref_mask[ref_mask < thresh] = 0.0
        ref_mask[ref_mask >= thresh] = 1.0

        up = (2 * ref_mask * pred_mask).sum()
        down = (ref_mask + pred_mask).sum()
        dice = up / down

        dice_sum += dice

        report.append([ref_mask_key, dice])
        print(f'For file {ref_mask_key} dice: {dice}')

    dice = dice_sum / (i+1)
    report.append(['Avg', dice])
    print(f'Avg dice: {dice}')

    report = [header] + report
    with open(eval_report_output, 'w') as f:
        csv_writer = csv.writer(f)
        for line in report:
            csv_writer.writerow(line)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    args = argparser.parse_args()

    config = read_config(args.config)

    main(config)
