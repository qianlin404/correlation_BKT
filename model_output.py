#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 10/8/18
# Description: This file include functions transfer model
#              output back to text files
# ========================================================

import pandas as pd
import numpy as np

from typing import Dict


def write_corr_matrix(corr_matrix: np.ndarray,
                      num_to_skill: Dict,
                      filename: str,
                      **kwargs):
    """
    Write corr matrix info file
    :param corr_matrix:
    :param nb_to_skill: dictionary convert numbers to string
    :return:
    """
    corr = pd.DataFrame(corr_matrix)
    skill_names = [num_to_skill[i] for i in range(corr_matrix.shape[0])]

    corr.index = skill_names
    corr.columns = skill_names

    corr.to_csv(filename, **kwargs)


def write_parameters(p: np.ndarray,
                     num_to_skill: Dict,
                     filename: str,
                     **kwargs):
    """
    Write models parameters to file
    :param p: model parameters
    :param num_to_skill: dictionary convert numbers to string
    :param filename:
    :return:
    """
    param = pd.DataFrame(p.reshape(-1, 1))
    skill_names = [num_to_skill[i] for i in range(param.shape[0])]

    param.index = skill_names
    param.columns = ["probability"]

    param.to_csv(filename, **kwargs)


def write_predictions(predictions: np.ndarray,
                      seq_length: np.ndarray,
                      filename: str,
                      input_data: np.ndarray,
                      num_to_student: Dict,
                      num_to_skill: Dict,
                      max_comp: int,
                      delimiter='~~',
                      **kwargs):
    """
    Append predictions to input data and convert them back to text
    :param predictions: Predictions from model
    :param seq_length: Sequence length
    :param filename: Filename
    :param input_data: Numerize input data
    :param num_to_student: Dictionary convert number to student
    :param num_to_skill: Dictionary convert number to skill
    :param max_comp: Maximum number of composition skill
    :return:
    """
    D = input_data.shape[2]
    if seq_length.dtype != np.int:
        seq_length = seq_length.astype(np.int)
    total_length = np.sum(seq_length)

    input_flat = np.empty(shape=(total_length, D))
    pred_flat = np.empty(shape=(total_length, 1))

    # Flat out the first dimension
    index = 0
    for i in range(input_data.shape[0]):
        input_flat[index:index+seq_length[i]] = input_data[i, :seq_length[i], :]
        pred_flat[index:index+seq_length[i]] = predictions[i, :seq_length[i], :]

        index = index + seq_length[i]

    result = pd.DataFrame(np.hstack([input_flat[:, :D-max_comp], pred_flat]),
                          columns=["student_names", "labels", "predictions"])
    result["student_names"] = result["student_names"].apply(lambda x: num_to_student[x])

    def skill_comp_to_name(comb):
        names = [num_to_skill[skill_id] for skill_id in comb if skill_id != -1]
        return delimiter.join(names)

    # Convert skills back to text
    input_skill = input_flat[:, D-max_comp:]
    skills = [skill_comp_to_name(comb) for comb in input_skill]
    result["skills"] = skills

    result.to_csv(filename, **kwargs)


def print_board(student_nb: int,
                train_skill_nb: int,
                correct_cnt: int,
                total_cnt: int,
                accuracy: float):
    """ Helper function to print result """

    result_format = "{:<25}{:<51}#"
    board = "#############################################################################"
    title = "#                           TRAINING COMPLETED                              #"
    student_nb_line = result_format.format("#  Evaluated student #:", student_nb)
    train_skill_nb_line = result_format.format("#  Train Skill #:", train_skill_nb)
    correct_cnt_line = result_format.format("#  Correct #:", correct_cnt)
    total_cnt_line = result_format.format("#  Totoal #:", total_cnt)
    accuracy_line = result_format.format("#  Accuracy #:", accuracy)

    print(board)
    print(title)
    print(student_nb_line)
    print(train_skill_nb_line)
    print(correct_cnt_line)
    print(total_cnt_line)
    print(accuracy_line)
    print(board)



