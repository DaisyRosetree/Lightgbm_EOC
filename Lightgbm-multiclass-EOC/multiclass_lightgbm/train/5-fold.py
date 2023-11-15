import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, \
    recall_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from config import Config
from multiclass_lightgbm.data_process.data_process import LightGBMDataLoader
import os
from multiclass_lightgbm.model.lightgbm_for_multi_label import train_lightgbm_model

data_path = "../dataset/2d_stroma-score_selected.csv"
target_column = "stroma score"
output_folder_path = "../output/K-fold/"
df_output_folder = '../output'

if __name__ == '__main__':
    config = Config()
    model_params = config.get_model_params()

    data_loader = LightGBMDataLoader(data_path, target_column, config)
    data_loader.load_and_preprocess_data()

    X_train_scaled = data_loader.X_train_scaled
    X_test_scaled = data_loader.X_test_scaled
    y_train = data_loader.y_train
    y_test = data_loader.y_test

    print(X_train_scaled)

    # Set random seeds
    random_state = 42
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    with tqdm(total=5, desc="Cross Validation Progress") as pbar:
        # Create an empty list to store performance metrics for each fold
        macro_precision_scores = []
        macro_recall_scores = []
        macro_f1_scores = []
        macro_auc_scores = []
        micro_precision_scores = []
        micro_recall_scores = []
        micro_f1_scores = []
        micro_auc_scores = []
        accuracy_scores = []

        # Create a dictionary to store the performance metrics for each fold
        fold_data = {
            'Fold': [],
            'Value': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'AUC': [],
            'Accuracy': []
        }

        for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
            X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Training
            model = train_lightgbm_model(X_train_fold, y_train_fold, model_params)

            # Make predictions on validation sets
            y_pred = model.predict(X_val_fold)
            y_pred_binary = y_pred.argmax(axis=1)

            # macro_precision
            macro_precision = precision_score(y_val_fold, y_pred_binary, average='macro')
            macro_precision_scores.append(macro_precision)

            # macro_recall
            macro_recall = recall_score(y_val_fold, y_pred_binary, average='macro')
            macro_recall_scores.append(macro_recall)

            # acro_f1
            macro_f1 = f1_score(y_val_fold, y_pred_binary, average='macro')
            macro_f1_scores.append(macro_f1)

            # label_binarizer
            label_binarizer = LabelBinarizer()
            y_val_bin = label_binarizer.fit_transform(y_val_fold)
            macro_auc = roc_auc_score(y_val_bin, y_pred, average='macro', multi_class='ovr')
            macro_auc_scores.append(macro_auc)

            # micro_precision
            micro_precision = precision_score(y_val_fold, y_pred_binary, average='micro')
            micro_precision_scores.append(micro_precision)

            # micro_recall
            micro_recall = recall_score(y_val_fold, y_pred_binary, average='micro')
            micro_recall_scores.append(micro_recall)

            # micro_f1
            micro_f1 = f1_score(y_val_fold, y_pred_binary, average='micro')
            micro_f1_scores.append(micro_f1)

            # micro_au
            micro_auc = roc_auc_score(y_val_bin, y_pred, average='micro', multi_class='ovr')
            micro_auc_scores.append(micro_auc)

            # accuracy
            accuracy = accuracy_score(y_val_fold, y_pred_binary)
            accuracy_scores.append(accuracy)

            print(f"Fold {i}:")
            print(f"macro_precision: {macro_precision:.4f}")
            print(f"macro_recall: {macro_recall:.4f}")
            print(f"macro_f1: {macro_f1:.4f}")
            print(f"macro_auc: {macro_auc:.4f}")
            print(f"micro_precision: {micro_precision:.4f}")
            print(f"micro_recall: {micro_recall:.4f}")
            print(f"micro_f1: {micro_f1:.4f}")
            print(f"micro_auc: {micro_auc:.4f}")
            print(f"accuracy: {accuracy:.4f}")
            print()

            metrics = ['           Precision', 'Recall', 'F1', 'AUC     ', '       Accuracy']
            macro_values = [macro_precision, macro_recall, macro_f1, macro_auc, accuracy]
            micro_values = [micro_precision, micro_recall, micro_f1, micro_auc, accuracy]

            Value = ['Macro Value', 'Micro Value']
            # Add performance metrics for each fold to the fold_data dictionary
            fold_data['Fold'].extend([i] * len(Value))
            fold_data['Value'].extend(Value)
            fold_data['Precision'].extend([macro_precision, micro_precision])
            fold_data['Recall'].extend([macro_recall, micro_recall])
            fold_data['F1'].extend([macro_f1, micro_f1])
            fold_data['AUC'].extend([macro_auc, micro_auc])
            fold_data['Accuracy'].extend([accuracy, accuracy])

            # Ensure that the number of data points is consistent
            if len(metrics) == len(macro_values) == len(micro_values):

                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]

                macro_values += [macro_values[0]]
                micro_values += [micro_values[0]]
                metrics += [metrics[0]]

                fig, ax = plt.subplots(subplot_kw={'polar': True})  # 创建极坐标轴
                ax.tick_params(axis='y', labelsize=16, labelcolor='gray')

                # Plot
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

                ax.set_title(f'Fold-{i}', fontsize=24)
                plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=18)
            else:
                print("The number of data points is inconsistent and cannot be mapped")

            dpi = 300
            pdf_width = 9.8
            pdf_height = 8
            plt.gcf().set_size_inches(pdf_width, pdf_height)
            file_name = f'radar_chart_{i}.pdf'
            file_path = os.path.join(output_folder_path, file_name)
            plt.savefig(file_path, format='pdf', dpi=dpi)

            pbar.update(1)

    # macro_precision
    mean_macro_precision = np.mean(macro_precision_scores)
    se_macro_precision = np.std(macro_precision_scores) / np.sqrt(len(macro_precision_scores))
    confidence_interval_macro_precision = 1.96 * se_macro_precision
    lower_bound_macro_precision = mean_macro_precision - confidence_interval_macro_precision
    upper_bound_macro_precision = mean_macro_precision + confidence_interval_macro_precision
    print(f"mean_macro_precision: {mean_macro_precision:.4f}")
    print(f"95% CI for macro_precision: ({lower_bound_macro_precision:.4f}, {upper_bound_macro_precision:.4f})")

    # macro_recall
    mean_macro_recall = np.mean(macro_recall_scores)
    se_macro_recall = np.std(macro_recall_scores) / np.sqrt(len(macro_recall_scores))
    confidence_interval_macro_recall = 1.96 * se_macro_recall
    lower_bound_macro_recall = mean_macro_recall - confidence_interval_macro_recall
    upper_bound_macro_recall = mean_macro_recall + confidence_interval_macro_recall
    print(f"mean_macro_recall: {mean_macro_recall:.4f}")
    print(f"95% CI for macro_recall: ({lower_bound_macro_recall:.4f}, {upper_bound_macro_recall:.4f})")

    # macro_f1
    mean_macro_f1 = np.mean(macro_f1_scores)
    se_macro_f1 = np.std(macro_f1_scores) / np.sqrt(len(macro_f1_scores))
    confidence_interval_macro_f1 = 1.96 * se_macro_f1
    lower_bound_macro_f1 = mean_macro_f1 - confidence_interval_macro_f1
    upper_bound_macro_f1 = mean_macro_f1 + confidence_interval_macro_f1
    print(f"mean_macro_f1: {mean_macro_f1:.4f}")
    print(f"95% CI for macro_f1: ({lower_bound_macro_f1:.4f}, {upper_bound_macro_f1:.4f})")

    # macro_auc
    mean_macro_auc = np.mean(macro_auc_scores)
    se_macro_auc = np.std(macro_auc_scores) / np.sqrt(len(macro_auc_scores))
    confidence_interval_macro_auc = 1.96 * se_macro_auc
    lower_bound_macro_auc = mean_macro_auc - confidence_interval_macro_auc
    upper_bound_macro_auc = mean_macro_auc + confidence_interval_macro_auc
    print(f"mean_macro_auc: {mean_macro_auc:.4f}")
    print(f"95% CI for macro_auc: ({lower_bound_macro_auc:.4f}, {upper_bound_macro_auc:.4f})")

    # micro_precision
    mean_micro_precision = np.mean(micro_precision_scores)
    se_micro_precision = np.std(micro_precision_scores) / np.sqrt(len(micro_precision_scores))
    confidence_interval_micro_precision = 1.96 * se_micro_precision
    lower_bound_micro_precision = mean_micro_precision - confidence_interval_micro_precision
    upper_bound_micro_precision = mean_micro_precision + confidence_interval_micro_precision
    print(f"mean_micro_precision: {mean_micro_precision:.4f}")
    print(f"95% CI for micro_precision: ({lower_bound_micro_precision:.4f}, {upper_bound_micro_precision:.4f})")

    # micro_recall
    mean_micro_recall = np.mean(micro_recall_scores)
    se_micro_recall = np.std(micro_recall_scores) / np.sqrt(len(micro_recall_scores))

    confidence_interval_micro_recall = 1.96 * se_micro_recall

    lower_bound_micro_recall = mean_micro_recall - confidence_interval_micro_recall
    upper_bound_micro_recall = mean_micro_recall + confidence_interval_micro_recall
    print(f"mean_micro_recall: {mean_micro_recall:.4f}")
    print(f"95% CI for micro_recall: ({lower_bound_micro_recall:.4f}, {upper_bound_micro_recall:.4f})")

    # micro_f1
    mean_micro_f1 = np.mean(micro_f1_scores)
    se_micro_f1 = np.std(micro_f1_scores) / np.sqrt(len(micro_f1_scores))

    confidence_interval_micro_f1 = 1.96 * se_micro_f1

    lower_bound_micro_f1 = mean_micro_f1 - confidence_interval_micro_f1
    upper_bound_micro_f1 = mean_micro_f1 + confidence_interval_micro_f1
    print(f"mean_micro_f1: {mean_micro_f1:.4f}")
    print(f"95% CI for micro_f1: ({lower_bound_micro_f1:.4f}, {upper_bound_micro_f1:.4f})")

    # micro_auc
    mean_micro_auc = np.mean(micro_auc_scores)
    se_micro_auc = np.std(micro_auc_scores) / np.sqrt(len(micro_auc_scores))

    confidence_interval_micro_auc = 1.96 * se_micro_auc

    lower_bound_micro_auc = mean_micro_auc - confidence_interval_micro_auc
    upper_bound_micro_auc = mean_micro_auc + confidence_interval_micro_auc
    print(f"mean_micro_auc: {mean_micro_auc:.4f}")
    print(f"95% CI for micro_auc: ({lower_bound_micro_auc:.4f}, {upper_bound_micro_auc:.4f})")

    # accuracy
    mean_accuracy = np.mean(accuracy_scores)
    se_accuracy = np.std(accuracy_scores) / np.sqrt(len(accuracy_scores))

    confidence_interval_accuracy = 1.96 * se_accuracy

    lower_bound_accuracy = mean_accuracy - confidence_interval_accuracy
    upper_bound_accuracy = mean_accuracy + confidence_interval_accuracy
    print(f"mean_accuracy: {mean_accuracy:.4f}")
    print(f"95% CI for accuracy: ({lower_bound_accuracy:.4f}, {upper_bound_accuracy:.4f})")

    # Create a dictionary to store statistics
    data = {
        'Metric': ['Macro Precision', 'Macro Recall', 'Macro F1', 'Macro AUC', 'Micro Precision', 'Micro Recall',
                   'Micro F1', 'Micro AUC', 'Accuracy'],
        'Mean Value': [mean_macro_precision, mean_macro_recall, mean_macro_f1, mean_macro_auc, mean_micro_precision,
                       mean_micro_recall, mean_micro_f1, mean_micro_auc, mean_accuracy],
        'Lower Bound': [lower_bound_macro_precision, lower_bound_macro_recall, lower_bound_macro_f1,
                        lower_bound_macro_auc, lower_bound_micro_precision, lower_bound_micro_recall,
                        lower_bound_micro_f1, lower_bound_micro_auc, lower_bound_accuracy],
        'Upper Bound': [upper_bound_macro_precision, upper_bound_macro_recall, upper_bound_macro_f1,
                        upper_bound_macro_auc, upper_bound_micro_precision, upper_bound_micro_recall,
                        upper_bound_micro_f1, upper_bound_micro_auc, upper_bound_accuracy]
    }

    fold_df = pd.DataFrame(fold_data)

    fold_df = fold_df.round(4)

    output_file_fold = 'performance_metrics_fold.xlsx'
    output_path = os.path.join(df_output_folder, output_file_fold)

    writer_fold = pd.ExcelWriter(output_path, engine='openpyxl')
    fold_df.to_excel(writer_fold, sheet_name='Performance Metrics Fold', index=False)
    writer_fold.save()

    df = pd.DataFrame(data)
    df = df.round(4)

    output_file_all = 'performance_metrics_all.xlsx'
    output_path = os.path.join(df_output_folder, output_file_all)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    df.to_excel(writer, sheet_name='Performance Metrics', index=False)

    writer.save()
