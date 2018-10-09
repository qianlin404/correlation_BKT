#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/27/18
# Description: Input pipe for BKT model
# ========================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json

from typing import List, Tuple, Dict
from itertools import chain

STUDENT_IDX = 0
SKILL_IDX = 3
TIME_IDX = 2
LABEL_IDX = 1

def parse_data_to_number(filename: str,
                         file_delimiter: str,
                         skill_delimiter: str,
                         output_dir,
                         max_length: np.int = None,
                         skill_set: Dict = None,
                         student_set: Dict = None,
                         max_composition: int = None):
    """
    Parse input data to numbers
    :param filename: Data filename
    :param file_delimiter: Delimiter to seperate columns
    :param max_length: The maximum sequence length allow for each sample
    :param skill_delimiter: Delimiter to split skills
    :param skill_set: Exists skill set to map
    :return input_data: Dataset contains only numbers
    """

    # Read data from text, the data should not have header and the columns
    # shoule be in order [student_id, label, time, skill_name]
    raw_data = pd.read_csv(filename, sep=file_delimiter, header=None)

    # Fill unknown skill and find skill sets
    num_null = np.sum(raw_data[SKILL_IDX].isnull())
    raw_data[SKILL_IDX] = raw_data[SKILL_IDX].fillna("UNKNOWN")

    skill_set, skill_nb = find_skill_sets(raw_data, delimiter=skill_delimiter, skill_set=skill_set)

    # Find student sets
    student_set, student_nb = find_student_sets(raw_data, student_set=student_set)

    # Split compositional skills to single skills, one for each columns
    input_data, max_composition = split_skills(raw_data, skill_delimiter, max_composition=max_composition)

    # Map dataset to numbers
    input_data = to_numbers(input_data, student_set, skill_set)
    input_data, seq_length = cut_sequences(input_data, sequence_length=max_length)

    # Collect result
    meta_data = {"skill_nb": skill_nb,
                 "student_nb": student_nb,
                 "max_sequence_length": max(seq_length),
                 "avg_sequence_length": float(np.mean(seq_length)),
                 "max_composition": int(max_composition),
                 "skill_null_nb": int(num_null)}
    result = {"input_data": input_data,
              "seq_length": seq_length,
              "student_set": student_set,
              "skill_set": skill_set,
              "meta_data": meta_data}

    # Save data
    name = os.path.splitext(os.path.basename(filename))[0]

    with open(os.path.join(output_dir, name + "_meta.json"), 'w') as f:
        json.dump(meta_data, f)

    with open(os.path.join(output_dir, name + "_skill_set.json"), 'w') as f:
        json.dump(skill_set, f)

    with open(os.path.join(output_dir, name + "_student_set.json"), 'w') as f:
        json.dump(student_set, f)

    return result


def filter_dataset(df: pd.DataFrame,
                   features: List[str],
                   dropna_features: List[str]) -> pd.DataFrame:
    """
    Filter raw input dataset
    :param df: Raw input dataset
    :param features: Features left
    :param dropna_features: Drop rows with Nans in these features
    :return: Filtered dataset
    """
    df = df.dropna(subset=dropna_features)
    df = df[features]

    return df


def find_student_sets(df: pd.DataFrame,
                      student_set: Dict) -> Dict:
    """
    Get unique student sets and index
    :param df: Input dataset
    :param skill_set: Pre-define student set
    :return: Dictionary of students and unique student number
    """
    unique_students = np.unique(df[STUDENT_IDX].values)
    unique_students.sort()
    student_nb = len(unique_students)

    if not student_set:
        print("Creating student map for %d students" % len(unique_students))
        unique_students = {v: i for i, v in enumerate(unique_students)}
    else:
        unknown_students = [student_id for student_id in unique_students if student_id not in student_set.keys()]

        student_set_size = len(student_set)
        unique_students = student_set

        print("Adding %d students to student map" % len(unknown_students))
        for i, student_id in enumerate(unknown_students):
            unique_students[student_id] = student_set_size + i

    return unique_students, student_nb


def find_skill_sets(df: pd.DataFrame,
                    delimiter: str,
                    skill_set: Dict) -> Dict:
    """
    Get unique skill sets and index
    :param df: Input dataset
    :param skill_col_name: column name of skills
    :return: Dictionary of skills
    """
    if not skill_set:
        unique_skills = df[SKILL_IDX].apply(lambda x: x.split(delimiter))
        unique_skills = sorted(list(set(chain.from_iterable(unique_skills))))
        unique_skills = {v: i for i, v in enumerate(unique_skills)}
    else:
        unique_skills = skill_set

    skill_nb = len(unique_skills)
    return unique_skills, skill_nb


def split_skills(df: pd.DataFrame,
                 delimiter: str,
                 max_composition: int = None) -> pd.DataFrame:
    """
    Split compositional skill data to single skill data. This operation may extend
    columns
    :param df: Input data
    :param max_composition: Pre-define maximum composition
    :return: Dataframe with splited skills
    """
    if max_composition is None:
        max_composition = df[SKILL_IDX].apply(lambda x: len(x.split(delimiter))).max()

    skill_arr = np.full([df.shape[0], max_composition], '-1', dtype=np.object)

    for i, value in enumerate(df[SKILL_IDX].values):
        skills = value.split(delimiter)
        arr = np.full([max_composition], '-1', dtype=np.object)
        for j, s in enumerate(skills):
            if j < max_composition:
                arr[j] = s
        skill_arr[i] = arr

    column_names = ["skill%d" % i for i in range(max_composition)]

    skill_df = pd.DataFrame(data=skill_arr, columns=column_names, index=df.index)
    result = df.join(skill_df).drop(SKILL_IDX, axis=1)
    return result, max_composition


def to_numbers(df: pd.DataFrame,
               student_set: Dict,
               skill_set: Dict,
               logger=None) -> pd.DataFrame:
    """
    Convert student ID and Skill strings to numbers
    :param df: Input dataframe
    :param student_set: Unique student set
    :param skill_set: Unique skill set
    :return: result dataframe
    """
    def map_student(key):
        return student_set[key]

    def map_skill(key):
        if key == '-1':
            return -1

        try:
            return skill_set[key]
        except KeyError as e:
            # Map unseen skil to unknown
            msg = "Skill %s is not seen in training data, set it to UNKNOWN." % e
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return skill_set["UNKNOWN"]

    # Convert student ID to numbers
    df[STUDENT_IDX] = df[STUDENT_IDX].apply(map_student)

    # Convert skill to numbers
    column_names = df.filter(regex="skill[0-9]+", axis=1).columns.values

    for col in column_names:
        df[col] = df[col].apply(map_skill)

    return df


def cut_sequences(data: pd.DataFrame,
                  sequence_length: np.int = None) -> np.ndarray:
    """
    Filter and cut sequence to given length
    :param data: Input data
    :param sequence_length: Length of cut sequence
    :param student_col_name: col name of student ID
    :param sort_col_name: Sort sequence by
    :return: 3D array with shape student_size * sequence_length * feature_length
    """

    column_length = data.shape[1]
    data = data.set_index(STUDENT_IDX)
    student_sequence_length = data.groupby(data.index).size()
    student_ids = student_sequence_length.index.values
    student_nb = student_sequence_length.shape[0]

    if sequence_length:
        max_sequence_size = sequence_length
    else:
        max_sequence_size = np.max(student_sequence_length)

    def cut_one_sequence(seq: pd.DataFrame,
                         sequence_length: np.int) -> np.ndarray:
        """ Helper function to cut one sequence """
        student_seq_length = seq.shape[0]

        seq = seq.sort_values(TIME_IDX)
        seq = seq.drop(TIME_IDX, axis=1)

        result = seq.head(sequence_length).reset_index().values

        if sequence_length > student_seq_length:
            fill_matrix = np.zeros(shape=(sequence_length - student_seq_length, column_length - 1))
            result = np.concatenate([result, fill_matrix], axis=0)

        assert result.shape == (sequence_length, column_length-1), "Expect shape %s, but get %s" % ((sequence_length, column_length), result.shape)
        return result, min(student_seq_length, sequence_length)

    print("Cutting %d student sequences..." % len(student_sequence_length))
    cuted_sequences = np.empty([student_nb, max_sequence_size, column_length-1], dtype=np.int32)

    lengths = np.empty(student_nb)
    for i, student_id in enumerate(student_ids):
        cuted_sequences[i], length = cut_one_sequence(data.loc[[student_id]], max_sequence_size)
        lengths[i] = length

    return cuted_sequences, lengths

def read_corr_matrix(filename: str, sep: str) -> np.ndarray:
    """
    Read corr matrix from file
    :param filename:
    :return corr:
    """
    corr = pd.read_csv(filename, sep=sep, header=None)
    return corr


class Splitter:

    def __init__(self, data, n_fold):

        self._n_fold = n_fold
        data_size = data.shape[0]
        split_size = np.int(np.ceil(data_size / n_fold))

        data_folds = [data[i*split_size:(i+1)*split_size] for i in range(n_fold)]
        self._data_folds = data_folds

    def get_train_test_by_fold(self, fold_num):
        test_set = self._data_folds[fold_num]
        train_set = np.concatenate(self._data_folds[:fold_num] + self._data_folds[fold_num+1:], axis=0)

        return train_set, test_set


# Input pipe to extract, transform and load data
class InputPipe:

    def __init__(self, train: np.ndarray,
                 train_seq_length: np.ndarray,
                 test: np.ndarray,
                 test_seq_length: np.ndarray,
                 student_nb: np.int,
                 skill_nb: np.int,
                 max_composition: np.int,
                 batch_size: int = 30,
                 n_epoch=None,
                 seed=404):
        """
        Create data preprocessing pipeline
        :param train: trainning data
        :param train_seq_length: Sequence length for training data
        :param test: test data
        :param test_seq_length: Sequence length for test data
        :param student_nb: Number of unique students
        :param skill_nb: Number of unique skills
        :param max_composition: The max number of skill composition
        :param batch_size:
        :param n_epoch: Number of epoch, infinity if None
        :param n_fold: Number of fold for cross validation
        :param verbose: Print additional information
        """

        self._batch_size = batch_size
        self._student_nb = student_nb
        self._skill_nb = skill_nb
        self._max_composition = max_composition
        self._n_epoch = n_epoch
        self._seed = seed

        self._train = tf.convert_to_tensor(train)
        self._train_seq_length = tf.convert_to_tensor(train_seq_length)

        if test is not None:
            self._test = tf.convert_to_tensor(test)
            self._test_seq_length = tf.convert_to_tensor(test_seq_length)
        else:
            self._test = test

    def _one_hot_encoder_student(self, student_seq: tf.Tensor) -> tf.Tensor:
        """ Mapper function to map student number to one hot tensor """
        one_hot = tf.one_hot(student_seq, depth=self._student_nb)
        one_hot = tf.reshape(one_hot, [-1, self._student_nb])

        return one_hot

    def _one_hot_encoder_skill(self, skill_seq: tf.Tensor) -> tf.Tensor:
        """ Mapper function to map skill to one hot tensor """
        one_hot = tf.one_hot(skill_seq, depth=self._skill_nb)
        one_hot = tf.reduce_sum(one_hot, axis=1)

        return one_hot

    def _one_hot_encoder(self, student_seq: tf.Tensor,
                         skill_seq: tf.Tensor,
                         label_seq: tf.Tensor,
                         sequence_length: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Mapper function to do one hot encoding """
        with tf.name_scope("One_hot_encoder"):
            student_encode = self._one_hot_encoder_student(student_seq)
            skill_encode = self._one_hot_encoder_skill(skill_seq)
            label_seq = tf.cast(label_seq, tf.float32)

        return (student_encode, skill_encode, label_seq, sequence_length)

    def generate_train_test(self) -> Tuple[tf.data.Iterator]:
        """
        Generate training and test sets
        :param fold_num: Fold number use as test set
        :return: training set and test set
        """

        test_iter = None

        with tf.name_scope("Encode_and_Batch"):
            # 2 features and 1 label
            train_student, train_label, train_skill = tf.split(self._train, [1, 1, self._max_composition], axis=2)
            train = tf.data.Dataset.from_tensor_slices((train_student, train_skill, train_label,self._train_seq_length))

            # Using 200 as buffer size since our total number of student is ~517
            # 200 should be enough
            # TODO: Make this parameter as a FLAG
            train = train.shuffle(buffer_size=200, seed=self._seed)\
                         .repeat(count=self.n_epoch)\
                         .map(self._one_hot_encoder, num_parallel_calls=4)\
                         .batch(batch_size=self.batch_size, drop_remainder=True)\
                         .prefetch(self._batch_size)

            train_iter = train.make_one_shot_iterator()

            if self._test is not None:
                test_student, test_label, test_skill = tf.split(self._test, [1, 1, self._max_composition], axis=2)
                test = tf.data.Dataset.from_tensor_slices((test_student, test_skill, test_label, self._test_seq_length))
                test = test.map(self._one_hot_encoder, num_parallel_calls=4) \
                    .batch(batch_size=1, drop_remainder=True)  # Make the batch size fixed
                test_iter = test.make_one_shot_iterator()

        return train_iter, test_iter

    @property
    def n_epoch(self):
        return self._n_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def fold_num(self):
        return self._fold_num if self._fold_num else np.nan



