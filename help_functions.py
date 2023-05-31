import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score
from scipy.special import softmax



def count_trainable_parameters(model):
    """
    Count the number of trainable model parameters.

    :param model: torch model instance
    :return: the number of trainable model parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    """
    Count the number of all model parameters.

    :param model: torch model instance
    :return: the number of trainable model parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_stats(outputs, y, acc_threshold=None):
    """
    Get statistics for given model outputs and true predictions 'y'.

    :param outputs: list of model predictions, each element of the list has a size (batch_size, 2)
    :param y: tensor of true labels (size equals to number of samples)
    :param acc_threshold: optional parameter to set threshold for a second neuron to improve model accuracy
    :return: (accuracy, AUROC, AUPRC)
    """
    outputs = torch.cat(outputs, dim=0)
    probs = torch.squeeze(torch.softmax(outputs, dim=1))
    probs = probs.cpu().detach().numpy()
    true = y.cpu().detach().numpy()
    try:
        auroc = roc_auc_score(true, probs[:, 1])
        auprc = average_precision_score(true, probs[:, 1])

        # todo: delete below 2 lines when not mixing tasks from different domains
        auroc = -1
        auprc = -1
    except ValueError as ve:  # for multi-class classification we calculate only accuracy
        auroc = -1
        auprc = -1

    if acc_threshold is None:
        predicted = np.argmax(outputs.cpu().detach().numpy(), axis=1).ravel()
        acc = np.sum(true == predicted) / true.shape[0]
    else:
        softmax_outputs_1 = softmax(outputs.cpu().detach().numpy(), axis=1)[:, 1]
        predicted = np.zeros(len(softmax_outputs_1))
        predicted[softmax_outputs_1 > acc_threshold] = 1
        acc = np.sum(true == predicted) / true.shape[0]

    return acc, auroc, auprc


def remove_empty_values(metric_epoch, num_tasks, num_epochs):
    """
    Remove empty values from 'metric_epoch', where there are zeros at some epochs (because of early stopping).

    :param metric_epoch: array of size (num_runs, num_tasks * num_epochs) with 0 values where early stopping occured
    :param num_tasks: number of tasks
    :param num_epochs: number of epochs per task
    :return: array of size (num_runs, num_tasks) with final metric value for each task in each run
    """
    epoch_no0 = []
    for row_i in range(len(metric_epoch)):
        row = metric_epoch[row_i]
        epoch_no0_row = []
        for task_i in range(num_tasks):
            index = ((task_i + 1) * num_epochs) - 1  # index of the last measured metric for each task
            while row[index] == 0:
                index -= 1
            epoch_no0_row.append(row[index])
        epoch_no0.append(epoch_no0_row)
    return epoch_no0


def get_model_outputs(model, batch_X, batch_mask, use_MLP, use_PSP, contexts, task_index):
    """
    Get model outputs with the forward pass of the batch of input data (batch_X).

    :param model: torch model instance
    :param batch_X: a batch of input data
    :param batch_mask: a batch of input data masks if used
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param use_PSP: boolean - if True, PSP method is used, meaning we need set of contexts for each task (including the first)
    :param contexts: contexts: binary context vectors
    :param task_index: index of the current task, which is being learned
    :return: batch model output
    """
    if use_PSP:
        if use_MLP:
            return model.forward(batch_X, use_PSP, contexts, task_index)
        else:
            return model.forward(batch_X, batch_mask, use_PSP, contexts, task_index)
    else:
        if use_MLP:
            return model.forward(batch_X)
        else:
            return model.forward(batch_X, batch_mask)


def nearest_task_mean_sample(samples, true_task_IDs):
    """
    Find the nearest task mean samples across all tasks to the given 'samples'. Compare nearest means to true task IDs.

    :param samples: list of data samples
    :param true_task_IDs: list of true task IDs (len(samples) = len(true_task_IDs))
    :return: the share ([0, 1]) of correctly predicted task IDs
    """
    means = [torch.load('X_mean_1.pt'), torch.load('X_mean_2.pt'), torch.load('X_mean_3.pt'),
             torch.load('X_mean_4.pt'), torch.load('X_mean_5.pt'), torch.load('X_mean_6.pt')]

    predicted_task_IDs = []
    for sample in samples:
        min_distance = 10 ** 9
        min_ID = 0
        for i, m in enumerate(means):
            distance = np.sqrt(np.sum((sample.cpu().detach().numpy() - m.cpu().detach().numpy()) ** 2))   # L2 distance

            # distance = np.sum(abs(sample.cpu().detach().numpy() - m.cpu().detach().numpy()))   # L1 distance
            if distance < min_distance:
                min_distance = distance
                min_ID = i
        predicted_task_IDs.append(min_ID)

    # compare predicted to true task IDs
    correct = 0
    for pred, tru in zip(predicted_task_IDs, true_task_IDs):
        if pred == tru:
            correct += 1
    return correct / len(samples)


def get_task_names(mode):
    """
    Get list with the order of tasks based on the selected mode.

    :param mode: string to select mode, options: 'NLP first', 'CV first', 'mixed', 'Split CIFAR-100'
    :return: 2D list of task names in short
    """
    if mode == 'NLP first':
        task_names = [['HS', 'SA', 'S', 'SA_2', 'C', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5'],
                      ['C', 'HD', 'SA', 'HS', 'SA_2', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5'],
                      ['SA', 'S', 'HS', 'SA_2', 'HD', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5'],
                      ['HD', 'SA_2', 'SA', 'C', 'S', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5'],
                      ['SA', 'HS', 'C', 'SA_2', 'HD', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5']]
    elif mode == 'CV first':
        task_names = [['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'HS', 'SA', 'S', 'SA_2', 'C'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'C', 'HD', 'SA', 'HS', 'SA_2'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'SA', 'S', 'HS', 'SA_2', 'HD'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'HD', 'SA_2', 'SA', 'C', 'S'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'SA', 'HS', 'C', 'SA_2', 'HD']]
    elif mode == 'mixed':
        task_names = [['HS', 'CIF1', 'SA', 'CIF2', 'S', 'CIF3', 'SA_2', 'CIF4', 'C', 'CIF5'],
                      ['C', 'CIF1', 'HD', 'CIF2', 'SA', 'CIF3', 'HS', 'CIF4', 'SA_2', 'CIF5'],
                      ['SA', 'CIF1', 'S', 'CIF2', 'HS', 'CIF3', 'SA_2', 'CIF4', 'HD', 'CIF5'],
                      ['HD', 'CIF1', 'SA_2', 'CIF2', 'SA', 'CIF3', 'C', 'CIF4', 'S', 'CIF5'],
                      ['SA', 'CIF1', 'HS', 'CIF2', 'C', 'CIF3', 'SA_2', 'CIF4', 'HD', 'CIF5']]
    elif mode == 'fixed NLP first':
        task_names = [['HS', 'SA', 'S', 'SA_2', 'C', 'CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5']] * 5
    elif mode == 'fixed CV first':
        task_names = [['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'HS', 'SA', 'S', 'SA_2', 'C']] * 5
    elif mode == 'fixed mixed':
        task_names = [['HS', 'CIF1', 'SA', 'CIF2', 'S', 'CIF3', 'SA_2', 'CIF4', 'C', 'CIF5']] * 5
    elif mode == 'Split CIFAR-100':
        task_names = [['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'CIF6', 'CIF7', 'CIF8', 'CIF9', 'CIF10'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'CIF6', 'CIF7', 'CIF8', 'CIF9', 'CIF10'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'CIF6', 'CIF7', 'CIF8', 'CIF9', 'CIF10'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'CIF6', 'CIF7', 'CIF8', 'CIF9', 'CIF10'],
                      ['CIF1', 'CIF2', 'CIF3', 'CIF4', 'CIF5', 'CIF6', 'CIF7', 'CIF8', 'CIF9', 'CIF10']]

    return task_names


