# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict


def multilabel_accuracy_mse_based(y_pred, y_true, extra_info=None):
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    mse = np.mean(np.square(y_pred - y_true))
    mse = 1 - mse
    return {"mse_ accuracy": mse}

# From: https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
def accuracy_multilabel(y_pred, y_true, extra_info=None):
    temp = 0
    print(len(y_true), len(y_pred))


    for i in range(len(y_true)):
        print(len(y_true[i]), len(y_pred[i]))
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return {"accuracy": temp / len(y_true)}

def accuracy(predictions, targets, extra_info=None) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def exact_match(predictions, targets):
  """Computes whether the targets match predictions exactly."""
  return {"em": 100 * float(np.array_equal(targets, predictions))}


# This is copied from pet.
def group_exact_match(predictions, targets, extra_info):
    """Computes the average exact match(EM) score for predictions and targets 
    corresponding to each question id."""
    question_ids = [v["group"] for v in extra_info]
    unique_q_ids = set(question_ids)
    id_to_targets = defaultdict(list)
    id_to_predictions = defaultdict(list)
    for q_id, target, prediction in zip(question_ids, targets, predictions):
        id_to_targets[q_id].append(target)
        id_to_predictions[q_id].append(prediction)

    # Computing the average em score for over question ids.
    ems = []
    for q_id in question_ids:
        ems.append(exact_match(id_to_predictions[q_id], id_to_targets[q_id])["em"])
    return {"em": np.mean(ems)}


def f1_macro(predictions, targets, extra_info=None):
    return {"f1-macro": 100*f1_score(targets, predictions, average="macro")}


def f1(predictions, targets, extra_info=None):
    return {"f1": 100*f1_score(targets, predictions)}
