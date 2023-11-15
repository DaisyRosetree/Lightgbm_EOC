import os
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from config import Config
from multiclass_lightgbm.data_process.data_process import LightGBMDataLoader
from multiclass_lightgbm.evaluation.evaluation import calculate_macro_precision, calculate_macro_recall, \
    calculate_macro_f1, calculate_macro_auc, calculate_micro_precision, calculate_micro_recall, calculate_micro_f1, \
    calculate_micro_auc, calculate_accuracy
from multiclass_lightgbm.model.lightgbm_for_multi_label import train_lightgbm_model
from multiclass_lightgbm.visualization.visualization import plot_confusion_matrix_test, plot_confusion_matrix_train


data_path = "dataset/2d_stroma-score_selected.csv"
target_column = "stroma score"
output_folder_path = './output'

if __name__ == '__main__':
    config = Config()
    model_params = config.get_model_params()

    data_loader = LightGBMDataLoader(data_path, target_column, config)
    data_loader.load_and_preprocess_data()

    X_test_scaled = data_loader.X_test_scaled
    y_test = data_loader.y_test

    X_train_scaled = data_loader.X_train_scaled
    y_train = data_loader.y_train

    # Training
    model = train_lightgbm_model(X_train_scaled, y_train, model_params)

    model.save_model('./output/model/model.txt')

    # Prediction
    y_pred_proba = model.predict(X_test_scaled)
    y_pred_labels = y_pred_proba.argmax(axis=1)

    macro_precision = calculate_macro_precision(y_test, y_pred_labels)
    macro_recall = calculate_macro_recall(y_test, y_pred_labels)
    macro_f1 = calculate_macro_f1(y_test, y_pred_labels)
    macro_auc = calculate_macro_auc(y_test, y_pred_proba)
    micro_precision = calculate_micro_precision(y_test, y_pred_labels)
    micro_recall = calculate_micro_recall(y_test, y_pred_labels)
    micro_f1 = calculate_micro_f1(y_test, y_pred_labels)
    micro_auc = calculate_micro_auc(y_test, y_pred_proba)
    accuracy = calculate_accuracy(y_test, y_pred_labels)

    print(f"macro_precision: {macro_precision:.4f}")
    print(f"macro_recall: {macro_recall:.4f}")
    print(f"macro_f1: {macro_f1:.4f}")
    print(f"macro_auc: {macro_auc:.4f}")
    print(f"micro_precision: {micro_precision:.4f}")
    print(f"micro_recall: {micro_recall:.4f}")
    print(f"micro_f1: {micro_f1:.4f}")
    print(f"micro_auc: {micro_auc:.4f}")
    print(f"accuracy: {accuracy:.4f}")

    metrics = ['Precision', 'Recall', 'F1', 'AUC', 'Accuracy']
    macro_values = [macro_precision, macro_recall, macro_f1, macro_auc, accuracy]
    micro_values = [micro_precision, micro_recall, micro_f1, micro_auc, accuracy]

    if len(metrics) == len(macro_values) == len(micro_values):

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        macro_values += [macro_values[0]]
        micro_values += [micro_values[0]]
        metrics += [metrics[0]]

        fig, ax = plt.subplots(subplot_kw={'polar': True})
        ax.tick_params(axis='y', labelsize=16, labelcolor='gray')

        ax.plot(angles, macro_values, 'b', label='Macro')
        ax.plot(angles, micro_values, 'r', label='Micro')
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics, fontsize=24)

        label_offset = 0.027
        for angle, macro, micro in zip(angles, macro_values, micro_values):
            macro_label_offset = label_offset if macro >= micro else -label_offset
            micro_label_offset = -label_offset if macro >= micro else label_offset
            ax.text(angle, macro + macro_label_offset, f'{macro:.3f}', ha='center', va='bottom', fontsize=18)
            ax.text(angle, micro + micro_label_offset, f'{micro:.3f}', ha='center', va='top', fontsize=18)

        ax.yaxis.set_major_locator(FixedLocator(np.arange(0.5, 1.05, 0.05)))
        ax.set_ylim(0.8, 1)

        ax.set_title('Validation Cohort', fontsize=24)

        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=18)
    else:
        print("The number of data points is inconsistent and cannot be mapped")

    dpi = 300
    pdf_width = 9.8
    pdf_height = 8
    plt.gcf().set_size_inches(pdf_width, pdf_height)
    file_name = 'Validation_radar.pdf'

    file_path = os.path.join(output_folder_path, file_name)

    plt.savefig(file_path, format='pdf', dpi=dpi)

    # Prediction on the training set
    y_pred_proba_train = model.predict(X_train_scaled)

    y_pred_labels_train = y_pred_proba_train.argmax(axis=1)

    macro_precision = calculate_macro_precision(y_train, y_pred_labels_train)
    macro_recall = calculate_macro_recall(y_train, y_pred_labels_train)
    macro_f1 = calculate_macro_f1(y_train, y_pred_labels_train)
    macro_auc = calculate_macro_auc(y_train, y_pred_proba_train)
    micro_precision = calculate_micro_precision(y_train, y_pred_labels_train)
    micro_recall = calculate_micro_recall(y_train, y_pred_labels_train)
    micro_f1 = calculate_micro_f1(y_train, y_pred_labels_train)
    micro_auc = calculate_micro_auc(y_train, y_pred_proba_train)
    accuracy = calculate_accuracy(y_train, y_pred_labels_train)

    print("Training data:")
    print(f"macro_precision: {macro_precision:.4f}")
    print(f"macro_recall: {macro_recall:.4f}")
    print(f"macro_f1: {macro_f1:.4f}")
    print(f"macro_auc: {macro_auc:.4f}")
    print(f"micro_precision: {micro_precision:.4f}")
    print(f"micro_recall: {micro_recall:.4f}")
    print(f"micro_f1: {micro_f1:.4f}")
    print(f"micro_auc: {micro_auc:.4f}")
    print(f"accuracy: {accuracy:.4f}")

    data = {
        'Metric': ['Macro Precision', 'Macro Recall', 'Macro F1', 'Macro AUC', 'Micro Precision', 'Micro Recall',
                   'Micro F1', 'Micro AUC', 'Accuracy'],
        'Mean Value': [macro_precision, macro_recall, macro_f1, macro_auc, micro_precision,
                       micro_recall, micro_f1, micro_auc, accuracy]
    }

    df = pd.DataFrame(data)
    df = df.round(4)

    output_file_all = 'Training performance_metrics.xlsx'
    output_path = os.path.join(output_folder_path, output_file_all)

    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    df.to_excel(writer, sheet_name='Training Performance Metrics', index=False)
    writer.save()

    # Visualization
    file_name_validation = 'validation_cm.pdf'
    file_name_training = 'training_cm.pdf'

    file_path_validation = os.path.join(output_folder_path, file_name_validation)
    file_path_training = os.path.join(output_folder_path, file_name_training)

    classes = ['0', '1', '2', '3']
    plot_confusion_matrix_test(y_test, y_pred_labels, classes)
    dpi = 300
    pdf_width = 8
    pdf_height = 6
    plt.gcf().set_size_inches(pdf_width, pdf_height)

    file_path = os.path.join(output_folder_path, file_name_validation)
    plt.savefig(file_path, format='pdf', dpi=dpi)

    plot_confusion_matrix_train(y_train, y_pred_labels_train, classes)
    plt.savefig(file_path_training, format='pdf')

    # bar chart
    file_name_bar = 'test_bar.pdf'
    file_path_bar = os.path.join(output_folder_path, file_name_bar)
    n_samples, n_classes = y_pred_proba.shape

    plt.figure(figsize=(12, 12))
    colors = ["#E7B800", "#2E9FDF", "#FF5733", "#4CAF50"]

    # Create a new index, sorting by category tags
    sorted_indices = np.argsort(y_test)

    # Use the sorted index to rearrange the prediction probability matrix
    y_pred_proba_sorted = y_pred_proba[sorted_indices]
    n_samples_per_class = [len(y_test[y_test == class_index]) for class_index in range(n_classes)]
    print(n_samples_per_class)

    # Plot a probability histogram for each category
    for class_index in range(n_classes):
        plt.bar(np.arange(n_samples), y_pred_proba_sorted[:, class_index], color=colors[class_index], alpha=0.5,
                label=f'Stroma score:  {class_index}')
        if class_index < n_classes - 1:
            x_position = sum(n_samples_per_class[:class_index + 1])
            plt.axvline(x=x_position, linestyle='--', color='gray')

    x_positions = np.cumsum(n_samples_per_class) - (np.array(n_samples_per_class) / 2)
    plt.xticks(x_positions, labels=[f'Score:{i}' for i in range(n_classes)], fontsize=34)
    plt.xlabel('Actual Stroma Score', fontsize=34)
    plt.ylabel('Predicted Probabilities', fontsize=34)
    plt.tick_params(axis='y', labelsize=34)
    plt.legend(loc='upper right', fontsize=28)
    plt.title('Predicted Probabilities for Each Patients', fontsize=34)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(file_path_bar, format='pdf')

    # Create a TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    max_display = 20
    class_names = ['stroma score = 0', 'stroma score = 1', 'stroma score = 2', 'stroma score = 3']

    class_shap_sum = [abs(shap_values[i]).sum() for i in range(len(shap_values))]
    sorted_indices = sorted(range(len(class_shap_sum)), key=lambda i: class_shap_sum[i], reverse=True)
    sorted_shap_values = [shap_values[i] for i in sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]

    plt.figure(figsize=(12, 12))
    shap.summary_plot(sorted_shap_values, X_train_scaled, max_display=max_display, class_names=class_names, show=False)

    fig = plt.gcf()

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)
    fig.set_size_inches(12, 8)
    legend = plt.legend()

    for text in legend.get_texts():
        text.set_fontsize(20)

    plt.savefig("./output/summary_plot.pdf", format='pdf')
    plt.show()

    plt.figure(figsize=(12, 12))
    shap.summary_plot(shap_values[1], X_test_scaled, class_names=class_names, show=False)
    fig = plt.gcf()

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)

    file_name_shap = f'shap_{1}.pdf'
    file_path_shap = os.path.join(output_folder_path, file_name_shap)

    plt.tight_layout()
    plt.savefig(file_path_shap, format='pdf')
    plt.show()











