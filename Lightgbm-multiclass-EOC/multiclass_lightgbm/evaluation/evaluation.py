from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer


def calculate_macro_precision(y_test, y_pred_labels):
    return precision_score(y_test, y_pred_labels, average='macro')


def calculate_macro_recall(y_test, y_pred_labels):
    return recall_score(y_test, y_pred_labels, average='macro')


def calculate_macro_f1(y_test, y_pred_labels):
    return f1_score(y_test, y_pred_labels, average='macro')


def calculate_macro_auc(y_test, y_pred_proba):
    label_binarizer = LabelBinarizer()
    y_true_bin = label_binarizer.fit_transform(y_test)
    return roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')


def calculate_micro_precision(y_test, y_pred_labels):
    return precision_score(y_test, y_pred_labels, average='micro')


def calculate_micro_recall(y_test, y_pred_labels):
    return recall_score(y_test, y_pred_labels, average='micro')


def calculate_micro_f1(y_test, y_pred_labels):
    return f1_score(y_test, y_pred_labels, average='micro')


def calculate_micro_auc(y_test, y_pred_proba):
    label_binarizer = LabelBinarizer()
    y_true_bin = label_binarizer.fit_transform(y_test)
    return roc_auc_score(y_true_bin, y_pred_proba, average='micro', multi_class='ovr')


def calculate_accuracy(y_test, y_pred_labels):
    return accuracy_score(y_test, y_pred_labels)