import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_test(y_true, y_pred, class_names):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Create a figure and axis for the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation Cohort', fontsize=20)

    # Set class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=20, rotation=45)
    plt.yticks(tick_marks, class_names, fontsize=20)

    # Display values in the cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=24
                     )

    plt.ylabel('Actual stroma-score', fontsize=20)
    plt.xlabel('Predicted stroma-score', fontsize=20)
    plt.tight_layout()


def plot_confusion_matrix_train(y_true, y_pred, class_names):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Create a figure and axis for the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Training Cohort', fontsize=20)

    # Set class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=20, rotation=45)
    plt.yticks(tick_marks, class_names, fontsize=20)

    # Display values in the cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=24)

    plt.ylabel('Actual stroma-score', fontsize=20)
    plt.xlabel('Predicted stroma-score', fontsize=20)
    plt.tight_layout()
