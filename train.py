#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 10/8/18
# Description: Train BKT model
# ========================================================

import os
import logging
import numpy as np
import tensorflow as tf

import input_pipe
import models
import model_output

def make_loger(logger_name: str,
               output_dir: str):
    log_format = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(output_dir, logger_name + ".log"))

    s_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.WARNING)

    s_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger


def train(train_filename: str,
          test_filename: str,
          file_delimiter: str,
          skill_delimiter: str,
          train_max_length: int,
          learning_init: float,
          slip_init: float,
          guess_init: float,
          transit_init: float,
          max_slip: float,
          max_guess: float,
          train_corr: bool,
          learning_rate: float,
          num_iter,
          batch_size,
          output_dir,
          num_epoch: int,
          corr_matrix_filename: str = None,
          seed=404
          ):
    """ Train and evaluate BKT model """
    train_logger = make_loger("Training", output_dir)

    train_logger.info("Parsing training data %s" % train_filename)

    # Load training and test data
    parse_train = input_pipe.parse_data_to_number(train_filename, output_dir=output_dir, file_delimiter=file_delimiter,
                                                  skill_delimiter=skill_delimiter, max_length=train_max_length)
    train_meta = parse_train["meta_data"]

    train_logger.info("Finished parsing training data, get %d students %d skills, maximum compositional skill is %d, total_length: %d"
                      % (train_meta["student_nb"], train_meta["skill_nb"],
                         train_meta["max_composition"], np.sum(parse_train["seq_length"])))

    if test_filename:
        train_logger.info("Parsing test data %s" % test_filename)
    else:
        train_logger.info("No test data found, using training set as test data")
        test_filename = train_filename

    parse_test = input_pipe.parse_data_to_number(test_filename, output_dir=output_dir, file_delimiter=file_delimiter,
                                                 skill_delimiter=skill_delimiter,
                                                 student_set=parse_train["student_set"],
                                                 skill_set=parse_train["skill_set"],
                                                 max_composition=train_meta["max_composition"])

    test_meta = parse_test["meta_data"]
    train_logger.info(
        "Finished parsing test data, get %d students %d skills, maximum compositional skill is %d, total_length: %d"
        % (test_meta["student_nb"], test_meta["skill_nb"],
           test_meta["max_composition"], np.sum(parse_test["seq_length"])))

    if train_corr and corr_matrix_filename:
        corr_matrix = input_pipe.read_corr_matrix(corr_matrix_filename, sep=file_delimiter)

        corr_shape = (train_meta["skill_nb"], train_meta["skill_nb"])
        if corr_matrix.shape != corr_shape:
            train_logger.error("Expect corr matrix shape (%d, %d), but get (%d, %d)" % (*corr_shape, *corr_matrix.shape))
            raise ValueError("Corr matrix shape must match skill number")
    else:
        corr_matrix = None

    # Begin model graph
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        train_logger.info("Constructing input pieline...")
        pipe = input_pipe.InputPipe(train=parse_train["input_data"], train_seq_length=parse_train["seq_length"],
                                    test=parse_test["input_data"], test_seq_length=parse_test["seq_length"],
                                    student_nb=train_meta["student_nb"], skill_nb=train_meta["skill_nb"],
                                    max_composition=train_meta["max_composition"], batch_size=batch_size,
                                    n_epoch=num_epoch, seed=seed)

        train_logger.info("Generating training iterator and test iterator set")
        train_iter, test_iter = pipe.generate_train_test()
        train = train_iter.get_next()
        test = test_iter.get_next()

        model = models.StandardBKTModel(skill_nb=train_meta["skill_nb"], max_guess=max_guess, max_slip=max_slip,
                                        train_corr_matrix=train_corr, corr_matrix=corr_matrix, learned_init=learning_init,
                                        transit_init=transit_init, slip_init=slip_init, guess_init=guess_init, log_dir=output_dir)

        model.train(train, train[2], iter_num=num_iter, global_step=global_step, learning_rate=learning_rate)

        train_logger.info("Predicting %d students, %d test set labels..." % (test_meta["student_nb"],
                                                                             np.sum(parse_test["seq_length"])))
        predictions = model.predict(test)

        transit_param = model.get_transit_parameters()
        slip_param = model.get_slip_parameters()
        guess_param = model.get_guess_parameters()
        if train_corr:
            corr_param = model.get_corr_matrix()

    acc, correct, total = models.get_accuracy(parse_test["input_data"][:, :, [1]], predictions,
                                              parse_test["seq_length"])

    model_output.print_board(test_meta["student_nb"], train_meta["skill_nb"], correct, total, np.round(acc, 4))

    train_logger.info("Writing result to %s..." % output_dir)

    num_to_skill = {num: skill for skill, num in parse_train["skill_set"].items()}
    num_to_student = {num: student for student, num in parse_test["student_set"].items()}

    model_output.write_predictions(predictions,
                                   parse_test["seq_length"],
                                   os.path.join(output_dir, "predictions.tsv"),
                                   parse_test["input_data"],
                                   num_to_student,
                                   num_to_skill,
                                   max_comp=train_meta["max_composition"],
                                   sep=file_delimiter
                                   )
    model_output.write_parameters(slip_param, num_to_skill, os.path.join(output_dir, "slip_param.tsv"), sep='\t')
    model_output.write_parameters(guess_param, num_to_skill, os.path.join(output_dir, "guess_param.tsv"), sep='\t')
    model_output.write_parameters(transit_param, num_to_skill, os.path.join(output_dir, "transit_param.tsv"), sep='\t')

    if train_corr:
        model_output.write_corr_matrix(corr_param, num_to_skill, os.path.join(output_dir, "corr.tsv"), sep='\t')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="BKT", description="Train and evaluate Standard Bayesian Knowledge Tracing model")
    parser.add_argument("train_filename", type=str, help="Train data file name")
    parser.add_argument("output_dir", type=str, help="model output directory")
    parser.add_argument("--test_filename", metavar='', type=str, default=None, help="Test data file name")
    parser.add_argument("--file_delimiter", metavar='', type=str, default='\t',
                        help="delimiter to parse all input files, default to tab")
    parser.add_argument("--skill_delimiter", metavar='', type=str, default='~~',
                        help="delimiter to parse compositional skills, default to \'~~\'")
    parser.add_argument("--train_max_length", metavar='', type=int, default=300,
                        help="max seqeunce length use for train, default to 300")
    parser.add_argument("--learning_init", metavar='', type=float, default=None,
                        help="initial learned probability for all skills, default initialize using uniform distribution")
    parser.add_argument("--slip_init", metavar='', type=float, default=None,
                        help="initial slip probability for all skills, default initialize using gaussian distribution")
    parser.add_argument("--guess_init", metavar='', type=float, default=None,
                        help="initial guess probability for all skills, default initialize using gaussian distribution")
    parser.add_argument("--transit_init", metavar='', type=float, default=None,
                        help="initial transit probability for all skills, default initialize using gaussian distribution")
    parser.add_argument("--max_slip", metavar='', type=float, default=.3,
                        help="max slip probability constraint, default to .3")
    parser.add_argument("--max_guess", metavar='', type=float, default=.3,
                        help="max guess probability constraint, default to .3")
    parser.add_argument("--train_corr", action="store_true",
                        help="flag to train the model using correlation matrix")
    parser.add_argument("--learning_rate", metavar='', type=float, default=1e-1,
                        help="learning rate, default to .1")
    parser.add_argument("--num_iter", metavar='', type=int, default=100,
                        help="number of iteration to train, default to 100")
    parser.add_argument("--batch_size", metavar='', type=int, default=30,
                        help="batch size for training, default to 30")
    parser.add_argument("--num_epoch", metavar='', type=int, default=10,
                        help="number of epoch to train, default to 10")
    parser.add_argument("--corr_matrix_filename", metavar='', type=str, nargs=1,
                        help="correlation matrix file name")

    args = parser.parse_args()

    train(train_filename=args.train_filename,
          test_filename=args.test_filename,
          file_delimiter=args.file_delimiter,
          skill_delimiter=args.skill_delimiter,
          train_max_length=args.train_max_length,
          learning_init=args.learning_init,
          slip_init=args.slip_init,
          guess_init=args.guess_init,
          transit_init=args.transit_init,
          max_slip=args.max_slip,
          max_guess=args.max_guess,
          train_corr=args.train_corr,
          learning_rate=args.learning_rate,
          num_iter=args.num_iter,
          batch_size=args.batch_size,
          output_dir=args.output_dir,
          num_epoch=args.num_epoch,
          corr_matrix_filename=args.corr_matrix_filename)
