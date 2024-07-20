#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json

from kunet_dk.train import main as kunet_training
from kunet_dk.gen_masks import main as kunet_gen_masks
from utils import read_config


def main(config_file_path):
    config = read_config(config_file_path)

    #from matplotlib import pyplot as plt
    #import numpy as np
    #x = np.linspace(0, 1, 100)
    #plt.plot(x, ((-200**-x)+1)*3-2)
    #plt.show()
    #pass

    if config['experiment_type'] == 'kunet_training':
        if config['script_purpose'] == 'training':
            kunet_training(config)
        elif config['script_purpose'] == 'gen_masks':
            kunet_gen_masks(config)

    exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    args = argparser.parse_args()
    main(args.config)
