#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/28/18
# Description: Models definition
# ========================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import logging

from typing import Callable
from datetime import datetime

def logit_fn(x: tf.Tensor):
    """ Inverse of sigmoid """
    return np.log(x/(1-x))

class KTRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 skill_nb: np.int32,
                 transit_initializer: Callable,
                 guess_initializer: Callable,
                 slip_initializer: Callable,
                 guess_max: tf.Tensor,
                 slip_max: tf.Tensor,
                 reuse=tf.AUTO_REUSE,
                 name=None,
                 dtype=tf.float32):
        """
        Initialization of BKTRNNCell
        :param skill_nb: Number of skills
        :param transit_initializer: Function use to initialize transit probability
        :param guess_initializer: Function use to initialize guess probability
        :param slip_initializer: Function use to initialize slip probability
        :param corr_matrix: Correlation matrix, if None, use identical matrix
        """
        super(KTRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

        self._skill_nb = skill_nb
        self._transit_initializer = transit_initializer
        self._guess_initializer = guess_initializer
        self._slip_initializer = slip_initializer

        self._guess_max = guess_max
        self._slip_max = slip_max

        self._corr_weights = tf.eye(skill_nb, dtype=tf.float32)

    def build(self, _):
        # Add transit weights
        with tf.variable_scope("Standard_BKT_parameters"):
            self._transit_weights = self.add_weight(name="transit_weights", shape=[1, self._skill_nb],
                                                    dtype=tf.float32, initializer=self._transit_initializer)

            # Add guess weights
            self._guess_weights = self.add_weight(name="guess_weights", shape=[1, self._skill_nb],
                                                  dtype=tf.float32, initializer=self._guess_initializer)

            # Add slip weights
            self._slip_weights = self.add_weight(name="slip_weights", shape=[1, self._skill_nb],
                                                 dtype=tf.float32, initializer=self._slip_initializer)

    def call(self, inputs, state):
        skills = inputs[1]
        observation = tf.reshape(inputs[2], [-1])

        # Extract variables
        # TODO: Try more combination functions
        with tf.name_scope("Standard_BKT_logic"):
            p_l = tf.reduce_sum(skills*state, axis=1)/tf.reduce_sum(skills, axis=1)
            p_t = tf.sigmoid(tf.reduce_sum(skills*self._transit_weights, axis=1))

            p_s = tf.sigmoid(tf.reduce_sum(skills*self._slip_weights, axis=1))
            p_s = self._slip_max * p_s

            p_g = tf.sigmoid(tf.reduce_sum(skills*self._guess_weights, axis=1))
            p_g = self._guess_max * p_g

            # Main logic
            output = p_l * (1 - p_s) + (1 - p_l) * p_g
            p_l_ob = observation * (p_l * (1 - p_s)) / (p_l * (1 - p_s) + (1 - p_l) * p_g) \
                     + (1 - observation) * (p_l * p_s) / (p_l * p_s + (1 - p_l) * (1 - p_g))

            p_l_next = p_l_ob + (1 - p_l_ob) * p_t

            delta = p_l_next - p_l
            corr = tf.map_fn(lambda x: tf.reduce_max(x*self._corr_weights, axis=1), skills)
            corr = corr * tf.reshape(delta, [-1, 1])

            new_state = tf.clip_by_value(state + corr, clip_value_min=tf.constant(.0), clip_value_max=tf.constant(1.0))
            output = tf.reshape(output, [-1, 1])  # Keep the same shape with observation
        return output, new_state

    @property
    def output_size(self):
        return 1

    @property
    def state_size(self):
        return self._skill_nb

    @property
    def p_slip(self):
        return tf.sigmoid(self._slip_weights) * self._slip_max

    @property
    def p_guess(self):
        return tf.sigmoid(self._guess_weights) * self._guess_max

    @property
    def p_trasit(self):
        return tf.sigmoid(self._transit_weights)


class Corr_KTRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 skill_nb: np.int32,
                 transit_initializer: Callable,
                 guess_initializer: Callable,
                 slip_initializer: Callable,
                 guess_max: tf.Tensor,
                 slip_max: tf.Tensor,
                 corr_matrix: tf.Tensor = None,
                 reuse=tf.AUTO_REUSE,
                 name=None,
                 dtype=tf.float32):
        """
        Initialization of BKTRNNCell
        :param skill_nb: Number of skills
        :param transit_initializer: Function use to initialize transit probability
        :param guess_initializer: Function use to initialize guess probability
        :param slip_initializer: Function use to initialize slip probability
        :param corr_matrix: Correlation matrix, if None, use identical matrix
        """
        super(Corr_KTRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

        self._skill_nb = skill_nb
        self._transit_initializer = transit_initializer
        self._guess_initializer = guess_initializer
        self._slip_initializer = slip_initializer

        self._guess_max = guess_max
        self._slip_max = slip_max

        if corr_matrix:
            self._corr_matrix = corr_matrix
        else:
            corr_matrix = np.ones(shape=(skill_nb, skill_nb)) * .1
            np.fill_diagonal(corr_matrix, .9999)
            self._corr_matrix = corr_matrix

        # self._corr_weights = tf.eye(skill_nb, dtype=tf.float32)

    def build(self, _):
        # Add transit weights
        with tf.variable_scope("Standard_BKT_parameters"):
            self._transit_weights = self.add_weight(name="transit_weights", shape=[1, self._skill_nb],
                                                    dtype=tf.float32, initializer=self._transit_initializer)
            # Add guess weights
            self._guess_weights = self.add_weight(name="guess_weights", shape=[1, self._skill_nb],
                                                  dtype=tf.float32, initializer=self._guess_initializer)
            # Add slip weights
            self._slip_weights = self.add_weight(name="slip_weights", shape=[1, self._skill_nb],
                                                 dtype=tf.float32, initializer=self._slip_initializer)
            # Add corrlation matrix
            self._corr_weights = self.add_weight(name="corr_weights",
                                                 shape=[self._skill_nb, self._skill_nb],
                                                 dtype=tf.float32,
                                                 initializer=tf.constant_initializer(logit_fn(self._corr_matrix)))

    def call(self, inputs, state):
        skills = inputs[1]
        observation = tf.reshape(inputs[2], [-1])

        # Extract variables
        # TODO: Try more combination functions
        with tf.name_scope("Standard_BKT_logic"):
            p_l = tf.reduce_sum(skills*state, axis=1)/tf.reduce_sum(skills, axis=1)
            p_t = tf.sigmoid(tf.reduce_sum(skills*self._transit_weights, axis=1))

            p_s = tf.sigmoid(tf.reduce_sum(skills*self._slip_weights, axis=1))
            p_s = self._slip_max * p_s

            p_g = tf.sigmoid(tf.reduce_sum(skills*self._guess_weights, axis=1))
            p_g = self._guess_max * p_g

            # Main logic
            output = p_l * (1 - p_s) + (1 - p_l) * p_g
            p_l_ob = observation * (p_l * (1 - p_s)) / (p_l * (1 - p_s) + (1 - p_l) * p_g) \
                     + (1 - observation) * (p_l * p_s) / (p_l * p_s + (1 - p_l) * (1 - p_g))

            p_l_next = p_l_ob + (1 - p_l_ob) * p_t

            delta = p_l_next - p_l
            corr = tf.map_fn(lambda x: tf.reduce_max(x*tf.sigmoid(self._corr_weights), axis=1), skills)
            corr = corr * tf.reshape(delta, [-1, 1])

            new_state = tf.clip_by_value(state + corr, clip_value_min=tf.constant(.0), clip_value_max=tf.constant(1.0))
            output = tf.reshape(output, [-1, 1])  # Keep the same shape with observation
        return output, new_state

    @property
    def output_size(self):
        return 1

    @property
    def state_size(self):
        return self._skill_nb

    @property
    def corr_matrix(self):
        return tf.sigmoid(self._corr_weights)

    @property
    def p_slip(self):
        return tf.sigmoid(self._slip_weights) * self._slip_max

    @property
    def p_guess(self):
        return tf.sigmoid(self._guess_weights) * self._guess_max

    @property
    def p_trasit(self):
        return tf.sigmoid(self._transit_weights)

# Standard bayesian knowledge tracing model
class StandardBKTModel:
    def __init__(self, skill_nb: np.int, max_guess: np.float, max_slip: np.float,
                 learned_init: np.float = None, transit_init: np.float = None,
                 guess_init: np.float = None, slip_init: np.float = None,
                 train_corr_matrix=True, corr_matrix=None, log_dir=None):
        """
        Initialize BKT model
        :param skill_nb: The number of skills
        :param max_guess: Max guess probability
        :param max_slip: Max slip probability
        :param leanred_init: Initial value of learned parameters
        :param transit_init: Initial value of transit probability
        :param guess_init: Initial value of guess probability
        :param slip_init: Initial value of slip probability
        """
        # TODO: Add value checker for parameters

        self._skill_nb = skill_nb
        self._max_guess = max_guess
        self._max_slip = max_slip
        self._learned_init = learned_init
        self._transit_init = transit_init
        self._guess_init = guess_init
        self._slip_init = slip_init
        self._train_corr = train_corr_matrix
        self._corr_matrix = corr_matrix

        # Create graph and session for this model
        self._sess = tf.Session()

        self._built = False

        # Configure logger
        self._logger = logging.getLogger("Standard_BKT_Model")
        self._logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(log_dir, "model.log"))
        stream_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.WARNING)

        log_format = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        stream_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)

        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)


    def _build(self):
        """ Build cell """
        skill_nb = self._skill_nb
        transit_initializer = get_param_initializer(self._transit_init)
        guess_initializer = get_param_initializer(self._guess_init)
        slip_initializer = get_param_initializer(self._slip_init)

        if self._train_corr:
            self._cell = Corr_KTRNNCell(skill_nb=skill_nb,
                                        transit_initializer=transit_initializer,
                                        guess_initializer=guess_initializer,
                                        slip_initializer=slip_initializer,
                                        guess_max=tf.constant(self._max_guess, dtype=tf.float32),
                                        slip_max=tf.constant(self._max_slip, dtype=tf.float32),
                                        corr_matrix=self._corr_matrix)
        else:
            self._cell = KTRNNCell(skill_nb=skill_nb,
                                   transit_initializer=transit_initializer,
                                   guess_initializer=guess_initializer,
                                   slip_initializer=slip_initializer,
                                   guess_max=tf.constant(self._max_guess, dtype=tf.float32),
                                   slip_max=tf.constant(self._max_slip, dtype=tf.float32))


        with tf.name_scope("Initialize_learned_prob"):
            if self._learned_init:
                self._init_state = tf.constant(self._learned_init, shape=[1, skill_nb], dtype=tf.float32)
            else:
                self._init_state = tf.random_uniform(shape=[1, skill_nb], maxval=1.0, minval=0.0,
                                               dtype=tf.float32)

        self._built = True

    def _predict_prob(self, inputs, sequence_length):
        """
        Predict probability
        :param inputs: Nested tuple (student_id, skills, observation) with shape [batch_size * sequence_length * n]
        :return: predicted observations with shape [batch_size * sequence_length * 1]
        """

        if not self._built:
            self._build()

        with tf.name_scope("predict_prob"):
            batch_size = inputs[0].shape[0]
            init_state = tf.tile(self._init_state, [batch_size, 1], name="Initialize_state")
            output, _ = tf.nn.dynamic_rnn(cell=self._cell, inputs=inputs, initial_state=init_state,
                                                   dtype=tf.float32, sequence_length=sequence_length)
        return output

    def _losses(self, labels, predictions):
        # Log loss
        return tf.losses.log_loss(labels=labels, predictions=predictions)

    def train(self, inputs, labels, iter_num, global_step=None, **kwargs):

        with tf.name_scope("train"):
            train_X = inputs[:3]
            train_seq_length = inputs[3]
            predictions = self._predict_prob(train_X, train_seq_length)
            loss = self._losses(labels, predictions)

            optimizer = tf.train.AdamOptimizer(**kwargs)
            train_op = optimizer.minimize(loss, global_step=global_step)
            correct_cnt_op, total_cnt_op = get_correct_and_total(labels, predictions, train_seq_length)

        tf.summary.scalar("losses", loss)
        tf.summary.scalar("train_acc", correct_cnt_op/total_cnt_op)
        tf.summary.histogram("pg", self._max_guess * tf.sigmoid(self._cell._guess_weights))
        tf.summary.histogram("pt", tf.sigmoid(self._cell._transit_weights))
        tf.summary.histogram("ps", self._max_slip * tf.sigmoid(self._cell._slip_weights))
        merged_summaries = tf.summary.merge_all()

        sub_fold = "corr_model" if self._train_corr else "standard_model"

        writer = tf.summary.FileWriter(os.path.join("tensorboard/", sub_fold), self._sess.graph)
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

        try:
            for i in range(iter_num):
                gs = self._sess.run(tf.train.get_global_step())
                _, summaries = self._sess.run([train_op, merged_summaries])

                losses, correct_cnt, total_cnt = self._sess.run([loss, correct_cnt_op, total_cnt_op])
                self._logger.info("Step: %d Train loss: %.4f; Train correct: %d; Train total: %d; Train acc: %.4f"
                                  % (gs, losses, correct_cnt, total_cnt, correct_cnt/total_cnt))

                writer.add_summary(summaries, global_step=gs)

        except tf.errors.OutOfRangeError:
            self._logger.warning("Run out of training data, stop!")
            pass

        writer.close()

    def predict(self, inputs, report_per_loop=5):
        predictions = []
        loop_cnt = 0
        start_time = datetime.now()
        while True:
            try:
                X = inputs[:3]
                seq_length = inputs[3]
                predictions.extend(self._sess.run(self._predict_prob(X, seq_length)))

                if loop_cnt % report_per_loop == 0:
                    time_delta = (datetime.now() - start_time).seconds
                    eff = time_delta / report_per_loop

                    self._logger.info("Finished predict %d students, average speed: %.2fs/student" % (loop_cnt, eff))
                    start_time = datetime.now()
                loop_cnt += 1

            except tf.errors.OutOfRangeError:
                break
        return np.array(predictions)

    def get_slip_parameters(self):
        return self._sess.run(self._cell.p_slip)

    def get_guess_parameters(self):
        return self._sess.run(self._cell.p_guess)

    def get_transit_parameters(self):
        return self._sess.run(self._cell.p_trasit)

    def get_corr_matrix(self):
        return self._sess.run(self._cell.corr_matrix) if self._train_corr else None


def get_param_initializer(init_value):
    """ Helper function to build parameter initializer for BKT parameters"""
    if init_value:
        return tf.constant_initializer(logit_fn(init_value), dtype=tf.float32)
    else:
        return tf.truncated_normal_initializer(dtype=tf.float32)


def get_correct_and_total(labels: tf.Tensor,
                          pred_prob: tf.Tensor,
                          seq_length: tf.Tensor):
    """
    Count correct predictions and total predictions
    :param labels: Ground true label
    :param pred_prob: predictions
    :param seq_length: sequence length
    :return correct, total:
    """
    labels_local = tf.cast(labels, tf.float32)
    pred_prob_local = tf.cast(pred_prob, tf.float32)
    seq_length_local = tf.cast(seq_length, tf.float32)
    pred_prob_local = tf.round(pred_prob_local)
    correct_cnt_raw = tf.reduce_sum(tf.cast(tf.equal(labels_local, pred_prob_local), tf.float32))

    total_cnt = tf.reduce_sum(seq_length_local)
    dim_sum = tf.cast(tf.shape(labels)[0]*tf.shape(labels)[1], tf.float32)
    padding_cnt = dim_sum - total_cnt

    correct_cnt = correct_cnt_raw - padding_cnt

    return correct_cnt, total_cnt

def get_accuracy(labels, predictions, seq_length):
    """
    Compute predict accuracy
    :param labels: True labels
    :param predictions: predictions
    :param seq_length:
    :return acc, correct, total:
    """
    assert labels.shape == predictions.shape
    pred_bin = np.round(predictions)
    correct = np.sum(labels == pred_bin)

    size = labels.shape[0]*labels.shape[1]
    total = np.sum(seq_length)
    padding_size = size - total

    correct = correct - padding_size
    acc = correct/total

    return acc, correct, total
