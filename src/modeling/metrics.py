from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle

def cal_multilabel_metrics(y_true, y_pred):
    ''' Compute micro/macro AUROC and AUPRC
    
    :param y_true: All the actual labels
    :type y_true: torch.Tensor
    :param y_pre: Probabilities of all the predicted labels
    :type y_pre: torch.Tensor
    
    :return report: wanted metrics
    :rtype: float

    '''
    
    y_true = y_true.cpu().detach().numpy().astype(np.int)
    y_pred = y_pred.cpu().detach().numpy().astype(np.float)

    macro_avg_prec = average_precision_score(y_true, y_pred, average = 'macro')
    micro_avg_prec = average_precision_score(y_true, y_pred, average = 'micro')
    
    micro_auroc = roc_auc_score(y_true, y_pred, average = 'micro')
    
    try:
        macro_auroc = roc_auc_score(y_true, y_pred, average = 'macro')
    except:
        macro_auroc = 0.0

    return macro_avg_prec, micro_avg_prec, macro_auroc, micro_auroc
    
    
def roc_curves(y_true, y_pred, labels, epoch=None, save_path='./experiments/'):
    '''Compute and plot the ROC Curves for each class, 
    also macro and micro. Save as a png image.
    
    :param y_true: The actual labels
    :type y_true: torch.Tensor
    :param y_pred: The probabilities of the predicted labels
    :type y_pred: torch.Tensor
    :param labels: The labels in SNOMED CT code
    :type labels: list
    :param epoch: Epoch in which the predictions are made
    :type epoch: int
    '''

    y_true = y_true.cpu().detach().numpy().astype(np.int)
    y_pred = y_pred.cpu().detach().numpy().astype(np.float)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # AUROC, fpr and tpr for each label
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(labels))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('ROC Curves')

    # Plotting micro and macro ROCs
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    # Plotting ROCs for each class
    for i in range(len(labels)):
        ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(labels[i], roc_auc[i]))

    # Adding labels and titles for plots
    ax1.plot([0, 1], [0, 1], 'k--'); ax2.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05]); ax2.set_ylim([0.0, 1.05])
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate'); ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", prop={'size': 8}); ax2.legend(loc="lower right", prop={'size': 6})
    
    fig.tight_layout()
    # Saving the plot 
    name = "roc-e{}.png".format(epoch) if epoch else "roc-test.png"
    
    plt.savefig(save_path + '/' + name, bbox_inches = "tight")
    plt.close(fig)