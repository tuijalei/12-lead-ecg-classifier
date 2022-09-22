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
    name = "roc-e{}.png".format(epoch)
    
    plt.savefig(save_path + '/' + name, bbox_inches = "tight")
    plt.close(fig)

    
def evaluate_predictions(test_data, pred_dir):
    ''' Compute evaluation metrics for predictions
    
    :param test_data: The location of the test data
    :type test_data: str
    :param pred_dir: The location of the predictions made from the test data
    :type pred_dir: str
    
    '''
    
    # Saving the evaluation history
    history = {}
    history['micro_avg_prec'] = 0
    history['micro_auroc'] = 0
    history['accuracy'] = 0
    history['micro_f1'] = 0
    
    # Find and load the test data
    test_files = pd.read_csv(test_data, usecols=['path']).values.tolist()
    test_files = [f for file in test_files for f in file]
    label_classes, labels = load_labels(test_files)
    
    print('Class labels found from the header files of the test data: ', end='')
    print(', '.join(label_classes))

    # Find and load the predictions made from the test data
    prediction_files = [os.path.join(pred_dir, file) for file in os.listdir(pred_dir) if file.endswith('.csv') and not file.startswith('.')]
    output_classes, pred_labels, prob_score = load_predictions(prediction_files)
    print('Class labels found from the predictions: ', end='')
    print(', '.join(output_classes))
    
    # Organize the labels and outputs
    classes, labels, pred_labels, prob_scores = organize_labels_outputs(label_classes, output_classes, labels, pred_labels, prob_score)
    
    print()
    
    ## EVALUATION METRICS ##
    micro_avg_prec = average_precision_score(labels, pred_labels, average = 'micro')
    history['micro_avg_prec'] = micro_avg_prec
    print('Micro Average Precision:', micro_avg_prec)
    
    micro_auroc = roc_auc_score(labels, pred_labels, average = 'micro')
    history['micro_auroc'] = micro_auroc
    print('Micro AUROC:', micro_auroc)
    
    accuracy = accuracy_score(labels, pred_labels)
    history['accuracy'] = accuracy
    print('Accuracy:', accuracy)
    
    micro_f1 = f1_score(labels, pred_labels, average = 'micro')
    history['micro_f1'] = micro_f1
    print('Micro F1-score:', micro_f1)
    
    # Save the metrics
    history_savepath = os.path.join(pred_dir, 'eval_history.pickle')
    with open(history_savepath, mode='wb') as file:
        pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
    

def load_labels(files):
    ''' Load the class labels (diagnoses) from the header files.
    These are the actual labels.
    
    Labels should have the following form:
    #Dx: label_1, label_2, label_3
    
    :param files: Testdata files
    :type files: list
    
    :return classes: Class labels found from the header files
    :return labels: One hot encoded class labels
    :rtypes: list, numpy.ndarray
    '''
    
    # Find the class labels from the header files of the test files
    header_labels = []
    for file in files:
        file = file.replace('.mat', '.hea')
        with open (file, 'r') as f:
            for line in f:
                if line.startswith('#Dx'):
                    dxs = [d.strip() for d in line.split(': ')[1].split(',')]
                    header_labels.append(dxs)
                    
    # Identify classes
    classes = set.union(*map(set, header_labels))
    classes = sorted(classes)
    num_classes = len(classes)
    num_files = len(files)
    
    # One hot encode the classes
    labels = np.zeros((num_files, num_classes), dtype=np.bool)
    for i in range(num_files):
        dxs = header_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels[i, j] = 1
    
    return classes, labels


def load_predictions(files):
    ''' Load the predictions from csv files. 
    
    The csv should have the following form:
    # Record_ID
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           0
    #        0.02,        0.31,        0.67
    
    :param files: Csv files of the predictions
    :type: list
    
    :return classes: Class labels from the predictions (#2 row)
    :return pred_labels: Predicted labels in binary form, 1 if True, 0 if False (#3 row)
    :return prob_scores: Probability score of each predicted label (#4 row)
    :rtypes: list, numpy.ndarray, numpy.ndarray
    '''
    
    num_files = len(files)
    
    class_labels = list()
    tmp_pred_labels = list()
    tmp_prob_scores = list()
    for i in range(num_files):
        with open (files[i], 'r') as f:
            for j, line in enumerate(f):
                splitted_row = [splitted_row.strip() for splitted_row in line.split(',')]
                
                # Load class labels (SNOMED CT Code)
                if j == 1:
                    labels_tmp = splitted_row
                    class_labels.append(labels_tmp)
                
                # Load predicted label(s) (binary)
                elif j == 2:
                    preds = list()
                    for num in splitted_row:
                        number = 1 if num == '1' else 0
                        preds.append(number)
                    tmp_pred_labels.append(preds)
                
                # Load probability scores (scalar)
                elif j == 3:
                    probs = list()
                    for prob in splitted_row:
                        score = float(prob) if is_number(prob) else 0
                        probs.append(score)
                        
                    tmp_prob_scores.append(probs)
    
    # Identify classes
    classes = set.union(*map(set, class_labels))
    classes = sorted(classes)
    num_classes = len(classes)
    
    # Use one-hot encoding for binary outputs and the same order for scalar outputs
    pred_labels = np.zeros((num_files, num_classes), dtype=np.bool)
    prob_scores = np.zeros((num_files, num_classes), dtype=np.float64)
    for i in range(num_files):
        dxs = class_labels[i]
        for k, dx in enumerate(dxs):
            j = classes.index(dx)
            pred_labels[i, j] = tmp_pred_labels[i][k]
            prob_scores[i, j] = tmp_prob_scores[i][k]

    return classes, pred_labels, prob_scores


def is_number(x):
    '''Check if the input is a number'''
    try:
        float(x)
        return True
    except ValueError:
        return False


def organize_labels_outputs(label_classes, output_classes, tmp_labels, tmp_pred_labels, tmp_prob_scores):
    ''' Organize the actual labels and the predicted ones for evaluation metrics to be computed.
    There may be difference between so need to possibly rearrange them.
    
    :param label_classes: The actual class labels
    :type label_classes: list
    :param output_classes: The predicted class labels
    :type output_classes: list
    :param tmp_labels: One hot encoded class labels
    :type tmp_labels: np.ndarray
    :param tmp_pred_labels: Predicted labels in binary form, 1 if True, 0 if False
    :type: tmp_pred_labels: np.ndarray
    :param tmp_prob_scores: Probability score of each predicted label 
    :type tmp_prob_scores: np.ndarray
    
    :return classes: All classes from either the header files or the predictions
    :return labels: Rearranged one hot encoded labels 
    :return pred_labels: Rearranged predicted labels
    :return prob_scores: Rearranged probability scores of predicted labels
    :rtypes: list, numpy.ndarray, numpy.ndarray, numpy.ndarray
    '''
    
    # Include all classes from either the header files or the predictions
    classes = sorted(set(label_classes) | set(output_classes))
    num_classes = len(classes)

    # All of these should have the same length
    assert(len(tmp_labels) == len(tmp_pred_labels) == len(tmp_prob_scores))
    num_recordings = len(tmp_labels)
    
    # Rearrange the columns of the labels and the outputs 
    # (predicted labels in binary form and probability scores) 
    # to be consistent with the order of the classes
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for k, dx in enumerate(label_classes):
        j = classes.index(dx)
        labels[:, j] = tmp_labels[:, k]
    
    pred_labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    prob_scores = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for k, dx in enumerate(output_classes):
        j = classes.index(dx)
        pred_labels[:, j] = tmp_pred_labels[:, k]
        prob_scores[:, j] = tmp_prob_scores[:, k]
    
    return classes, labels, pred_labels, prob_scores
   