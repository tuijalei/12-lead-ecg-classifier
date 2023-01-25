from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle

 
def cal_multilabel_metrics(y_true, y_pre, labels, threshold=0.5):
    ''' Compute micro/macro AUROC and AUPRC
    
    :param y_true: Actual class labels
    :type y_true: torch.Tensor
    :param y_pre: Logits of predictions
    :type y_pre: torch.Tensor
    :param labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    
    :return: wanted metrics
    :rtypes: float
    '''
    
    # Convert tensors to numpy and filter out empty classes
    true_labels, pre_prob, cls_labels = preprocess_labels(y_true, y_pre, labels)
    
    # -------- One-hot-endcode predicted labels --------

    pre_binary = np.zeros(pre_prob.shape, dtype=np.int32)
    
    # Find the index of the maximum value within the logits
    likeliest_dx = np.argmax(pre_prob, axis=1)

    # First, add the most likeliest diagnosis to the predicted label
    pre_binary[np.arange(true_labels.shape[0]), likeliest_dx] = 1

    # Then, add all the others that are above the decision threshold
    other_dx = pre_prob >= threshold

    pre_binary = pre_binary + other_dx
    pre_binary[pre_binary > 1.1] = 1
    pre_binary = np.squeeze(pre_binary) 
    
    # ---------------- Wanted metrics ----------------

    # -- Average precision score
    macro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'macro')
    micro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'micro')
    
    # -- AUROC score
    micro_auroc = roc_auc_score(true_labels, pre_prob, average = 'micro')
    macro_auroc = roc_auc_score(true_labels, pre_prob, average = 'macro')
    
    # -- PhysioNet Challenge 2021 score
    challenge_metric = physionet_challenge_score(true_labels, pre_binary, cls_labels)

    return macro_avg_prec, micro_avg_prec, macro_auroc, micro_auroc, challenge_metric

    
def preprocess_labels(y_true, y_pre, labels):
    ''' Convert tensor variables to numpy and check the positive
    class labels. If there's none, leave the columns out from
    actual labels, probability estimates and class labels used
    in the classification.
    
    :param y_true: Actual class labels
    :type y_true: torch.Tensor
    :param y_pre: Probability estimates of predicted labels
    :type y_pre: torch.Tensor
    
    :return true_labels, pred_prob: Converted and possibly filtered
                                    actual labels and probability estimates
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    
    # Actual labels from tensor to numpy
    true_labels = y_true.cpu().detach().numpy().astype(np.int32)   

    # Probability estimates from tensor to numpy
    pre_prob = y_pre.cpu().detach().numpy().astype(np.float32)
    
    # Find all the columnwise indexes where's there's no positive class
    null_idx = np.argwhere(np.all(true_labels[..., :] == 0, axis=0))

    # Drop the all-zero columns from actual labels, class labels 
    # used in the classification and probability estimates of predictions
    if any(null_idx):
        true_labels = np.delete(true_labels, null_idx, axis=1)
        pre_prob = np.delete(pre_prob, null_idx, axis=1)
        labels = np.delete(labels, null_idx)

    # There should be as many actual labels and probability estimates
    # as there are labels left
    assert true_labels.shape[1] == pre_prob.shape[1] == len(labels)
    
    return true_labels, pre_prob, labels


def physionet_challenge_score(y_true, y_pre, labels):
    ''' Compute the PhysioNet Challenge 2021 metric based on the actual and
    predicted labels. The scoring awards full credit to correct diagnoses and
    partial credit to misdiagnoses that result in similar treatments or
    outcomes as the true diagnosis as judged by the cardiologists.
        
    :param y_true: Actual class labels
    :type y_true: numpy.ndarray
    :param y_pre: One-hot-encoded predicted labels
    :type y_pre: numpy.ndarray
    :labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    
    :return: challenge metric
    :rtype: float
    '''    
    
    # Identify the weights and the SNOMED CT code for the sinus rhythm class 
    weights_file = os.path.join(os.getcwd(), 'data', 'physionet2021_weights.csv')
    sinus_rhythm = '426783006' 
    
    # -------- Load the scored classes and the weights for the Challenge metric --------
    
    # Load the csv file of the weights
    weights_df = pd.read_csv(weights_file, index_col=0)
    indexes = sum([row.split('|') for row in weights_df.index], [])
    columns = sum([row.split('|') for row in weights_df.columns], [])

    # Indexes and columns should have the same values in the weight file as it's an NxN matrix
    assert indexes == columns, 'Columns and indexes in the weight file don´t match'
   
    # They should be similar so let's take the class labels from one
    # Also, remove equivalent classes (the ones are from PhysioNet Challenge 2021)
    equivalent_classes = ['59118001', '63593006', '17338001', '164909002']
    classes = sorted(list(set(indexes) - set(equivalent_classes)))

    # Get weights out of the weight file
    physionet_weights = weights_df.values 
    
    # ------------- Reshape actual and predicted labels -------------

    num_classes = len(classes)
    true_labels = np.zeros((len(y_true), num_classes), dtype=np.bool_)
    binary_outputs = np.zeros((len(y_pre), num_classes), dtype=np.bool_)
    for i, dx in enumerate(labels):
        class_index = classes.index(dx)
        true_labels[:, class_index] = y_true[:, i]
        binary_outputs[:, class_index] = y_pre[:, i]

    # ------------- Challenge metric -------------
    challenge_metric = compute_challenge_metric(physionet_weights,
                                                true_labels,
                                                binary_outputs,
                                                classes,
                                                sinus_rhythm)

    return challenge_metric


def compute_modified_confusion_matrix(labels, outputs):
    '''Compute a modified confusion matrix for multi-class, multi-label tasks.
    
    :param labels: Actual class labels
    :type labels: numpy.ndarray
    :param outputs: One-hot-encoded predicted class labels
    :type outputs: numpy.ndarray
    
    :return A: Multi-class, multi-label confusion matrix
    :rtype: numpy.ndarray
    
    '''

    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization
    return A


def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    ''' Compute the evaluation metric for the Challenge.
    
    :param weights: Physionet Challenge weight for each label
    :type weights: numpy.ndarray
    :param labels: Actual class labels
    :type labels: numpy.ndarray
    :param outputs: One-hot-encoded predicted labels
    :type outputs: numpy.ndarray
    :param classes: Labels used in scoring as SNOMED CT Codes
    :type classes: list
    :param sinus_rhythm: SNOMED CT Code of sinus rhythm
    :type sinus_rhythm: str
    
    :return normalized_score: normalized challenge metric
    :rtype: float
    '''
    
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


def roc_curves(y_true, y_pre, labels, epoch=None, save_path='./experiments/'):
    '''Compute and plot the ROC Curves for each class, also macro and micro. Save as a png image.
    
    :param y_true: Actual labels
    :type y_true: torch.Tensor
    :param y_pred: Probability estimates of predicted labels
    :type y_pred: torch.Tensor
    :param labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    :param epoch: Epoch in which the predictions are made
    :type epoch: int
    '''

    # Convert tensors to numpy and filter out classes
    true_labels, pre_prob, cls_labels = preprocess_labels(y_true, y_pre, labels)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # AUROC, fpr and tpr for each label
    for i in range(len(cls_labels)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pre_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), pre_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Interpolate all ROC curves at these points to compute macro-average ROC area
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(cls_labels)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average the mean TPR and compute AUC
    mean_tpr /= len(cls_labels)
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('ROC Curves')

    # Plotting micro-average and macro-average ROC curves
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    # Plotting ROCs for each class
    for i in range(len(cls_labels)):
        ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(cls_labels[i], roc_auc[i]))

    # Adding labels and titles for plots
    ax1.plot([0, 1], [0, 1], 'k--'); ax2.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05]); ax2.set_ylim([0.0, 1.05])
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", prop={'size': 8}); ax2.legend(loc="lower right", prop={'size': 6})
    
    fig.tight_layout()
    # Saving the plot 
    name = "roc-e{}.png".format(epoch) if epoch else "roc-test.png"
    
    plt.savefig(save_path + '/' + name, bbox_inches = "tight")
    plt.close(fig) 