# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch

from models import  *
from helpers import DataLoader
from helpers import BaseRunner
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2019,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    logging.info(utils.format_arg_str(args))

    # Random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: {}".format(torch.cuda.device_count()))

    # Load data
    corpus_path = '{}/{}/Corpus.pkl'.format(args.path, args.dataset)
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = DataLoader.DataLoader(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    model = model.double()
    model.apply(model.init_weights)
    model.actions_before_train()
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    # Run model
    runner = BaseRunner.BaseRunner(args)
    logging.info('Test Before Training: ' + runner.print_res(model, corpus))
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(model, corpus)
    logging.info(os.linesep + 'Test After Training: ' + runner.print_res(model, corpus))

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='Chorus', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = DataLoader.DataLoader.parse_data_args(parser)
    parser = BaseRunner.BaseRunner.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_file_name = [init_args.model_name, args.dataset, str(args.random_seed), 'optimizer=' + args.optimizer,
                     'lr=' + str(args.lr), 'l2=' + str(args.l2), 'dropout=' + str(args.dropout)]
    if 'Chorus' in init_args.model_name:
        log_file_name += ['margin=' + str(args.margin), 'scale=' + str(args.lr_scale)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}.txt'.format(log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)
    utils.check_dir(args.log_file)
    utils.check_dir(args.model_path)

    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
