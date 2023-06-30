import numpy as np
import torch
import math
from help_functions import *


def random_binary_array(size):
    """
    Create an array of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: length of the created array
    :return: binary numpy array with values -1 or 1
    """
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = -1
    vec[vec >= 0] = 1
    return vec


def create_context_vectors(model, num_tasks, element_wise, use_PSP=False):
    """
    Create random binary context vectors for all model layers.
    Return together with layer dimension side, which is a list of 0 (first dimension taken for context size)
    and 1 (second dimension taken for context size).

    :param model: torch model instance
    :param num_tasks: number of tasks
    :param element_wise: boolean - if True, the number of context values in self attention part is the same as number of parameters
    :param use_PSP: boolean - if True, PSP method is used, meaning we need set of contexts for each task (including the first)
    :return: context_vectors (shape=(num_tasks-1, num of model layers)), layer_dimension (length=num of model layers)
    """
    context_vectors = []
    layer_dimension = []
    n = num_tasks if use_PSP else num_tasks - 1
    for t in range(n):    # our contexts only needed between tasks, i.e. len(contexts)=num_task-1
        task_contexts = []
        for name, params in model.named_parameters():
            if name.endswith('weight'):     # only weight, not bias
                if 'self_attn' not in name:     # FC layer
                    vector_size = params.size()[1]
                    if t == 0:
                        layer_dimension.append(1)
                else:   # not FC layer (e.g., Wq, Wk, Wv in multi-head attention)
                    if element_wise:
                        vector_size = params.size()[0] * params.size()[1]
                        if t == 0:
                            layer_dimension.append(2)
                    else:
                        vector_size = params.size()[0]
                        if t == 0:
                            layer_dimension.append(0)

                binary_context_vector = random_binary_array(vector_size)
                task_contexts.append(binary_context_vector)
        context_vectors.append(task_contexts)

    # # Ablate a single W inside a multi-head attention layer
    # for con_vec in context_vectors:
    #     con_vec[0][1024:2048] = np.ones(1024)

    return context_vectors, layer_dimension


def create_adapter_context_vectors(model, num_tasks, adapter_trainable_layers):
    """
    Create random binary context vectors for trainable adapter layers.
    Return together with layer dimension side, which is a list of 0 (first dimension taken for context size)
    and 1 (second dimension taken for context size).

    :param model: torch model instance
    :param num_tasks: number of tasks
    :param adapter_trainable_layers: list of strings with the names of trainable adapter layers
    :return: context_vectors (shape=(num_tasks-1, num of adapter layers)), layer_dimension (length=num of adapter layers)
    """
    context_vectors = []
    layer_dimension = []
    n = num_tasks - 1
    for t in range(n):  # our contexts only needed between tasks, i.e. len(contexts)=num_task-1
        task_contexts = []
        for name, params in model.named_parameters():
            # if name.endswith('weight') and "layer_norm" not in name and name not in adapter_trainable_layers:  # only weight, not bias

            if name.endswith('weight') and 'mlp' in name and name in adapter_trainable_layers:
                vector_size = params.size()[1]
                if t == 0:
                    layer_dimension.append(1)

                # if 'self_attn' not in name:  # FC layer
                #     vector_size = params.size()[1]
                #     if t == 0:
                #         layer_dimension.append(1)
                # else:  # not FC layer (e.g., Wq, Wk, Wv in multi-head attention)
                #     vector_size = params.size()[0]
                #     if t == 0:
                #         layer_dimension.append(0)

                binary_context_vector = random_binary_array(vector_size)

                task_contexts.append(binary_context_vector)
        context_vectors.append(task_contexts)

    return context_vectors, layer_dimension


def context_multiplication(model, contexts, layer_dimension, task_index, adapter_trainable_layers=None):
    """
    Perform context multiplication of parameters in model.

    :param model: torch model instance
    :param contexts: binary context vectors (shape=(num_tasks-1, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size), 1 (second dimension taken), 2 (element-wise)
    :param task_index: index of the current task, which has finished learning
    :param adapter_trainable_layers: list of strings with the names of trainable adapter layers
    :return: None (but model parameters are updated)
    """
    # Ablation study (layers 0 - 5)
    first_ablated_layer = -1    # -1 means no ablation
    last_ablated_layer = -1     # -1 means no ablation
    ablated_layers = list(range(first_ablated_layer, last_ablated_layer + 1))

    layer_index = 0
    for name, params in model.named_parameters():
        if adapter_trainable_layers is None:
            condition = name.endswith('weight')
        else:
            condition = name.endswith('weight') and 'mlp' in name and name in adapter_trainable_layers

            # condition = name.endswith('weight') and "layer_norm" not in name and name not in adapter_trainable_layers

            # # do not superimpose last layer (superimpose only the two layers in adapter module)
            # condition = name.endswith('weight') and 'bottleneck_mlp' in name and name in adapter_trainable_layers


        if condition:  # only weight, not bias
            if layer_index not in ablated_layers:   # if layer is not ablated
                with torch.no_grad():
                    if layer_dimension[layer_index] == 0:
                        context_matrix = torch.from_numpy(np.diag(contexts[task_index][layer_index]).astype(np.float32)).cuda()
                        new_params = torch.matmul(context_matrix, params)
                    elif layer_dimension[layer_index] == 1:
                        context_matrix = torch.from_numpy(np.diag(contexts[task_index][layer_index]).astype(np.float32)).cuda()
                        new_params = torch.matmul(params, context_matrix)
                    elif layer_dimension[layer_index] == 2:    # element-wise multiplication
                        context_matrix = torch.from_numpy(np.reshape(contexts[task_index][layer_index],
                                                                     newshape=(params.size()[0], params.size()[1])).astype(np.float32)).cuda()
                        new_params = params * context_matrix
                    else:
                        raise ValueError('Layer dimension must be 0, 1 or 2.')

                    params.copy_(new_params)

            layer_index += 1


def evaluate_results(model, contexts, layer_dimension, all_tasks_test_data, superposition, task_index, first_average,
                     use_MLP, batch_size, use_PSP=False, adapter_trainable_layers=None, acc_thresholds=None,
                     task_names_string=None):
    """
    Evaluate the results on test data with or without using superposition. Return accuracy, AUROC and AUPRC.

    :param model: torch model instance
    :param contexts: binary context vectors (shape=(num_tasks-1, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size) and 1 (second dimension taken)
    :param all_tasks_test_data: list of all test data [X_test, y_test, mask_test] until the current task index
    :param superposition: boolean - True, if superposition is used
    :param task_index: index of the current task, which is being learned
    :param first_average: string - show results on 'first' task only or the 'average' results until current task index
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param batch_size: batch size
    :param use_PSP: boolean - if True, PSP method is used, meaning we need set of contexts for each task (including the first)
    :param adapter_trainable_layers: list of strings with the names of trainable adapter layers
    :param acc_thresholds: optional list of parameters to set threshold for a second neuron to improve model accuracy
    :param task_names_string: the order of learning tasks: 'NLP first', 'CV first', 'mixed', 'fixed NLP first',
                              'fixed CV first', 'fixed mixed' or 'Split CIFAR-100'
    :return: accuracy, AUROC, AUPRC
    """
    if superposition:   # superposition used
        if first_average == 'first':    # not implemented for PSP
            # unfold network parameters to the first task
            for task_i in range(task_index - 1, -1, -1):
                context_multiplication(model, contexts, layer_dimension, task_i)

            # evaluate the model on the first task
            acc, auroc, auprc = evaluate_current_task(model, all_tasks_test_data, 0, use_MLP)

            # restore model parameters to the old ones (before context multiplication)
            for task_i in range(task_index):
                context_multiplication(model, contexts, layer_dimension, task_i)

            return acc, auroc, auprc
        elif first_average == 'average':
            if use_PSP:
                return evaluate_tasks_average_PSP(model, all_tasks_test_data, contexts, layer_dimension, task_index, use_MLP)
            else:
                return evaluate_tasks_average(model, all_tasks_test_data, contexts, layer_dimension, superposition, task_index,
                                              use_MLP, batch_size, adapter_trainable_layers, acc_thresholds, task_names_string)
        else:
            raise ValueError('The value of "first_average" has to be string "first" or "average".')
    else:   # superposition not used
        if first_average == 'first':
            return evaluate_current_task(model, all_tasks_test_data, 0, use_MLP)
        elif first_average == 'average':
            return evaluate_tasks_average(model, all_tasks_test_data, contexts, layer_dimension,
                                          superposition, task_index, use_MLP, batch_size)
        else:
            raise ValueError('The value of "first_average" has to be string "first" or "average".')


def evaluate_current_task(model, all_tasks_test_data, task_index, use_MLP, acc_threshold=None):
    """
    Evaluate results on the first task, using the current model.

    :param model: torch model instance
    :param all_tasks_test_data: list of all test dataloaders ([X_test, y_test, mask_test]) until the current task index
    :param task_index: index of the current task
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param acc_threshold: optional parameter to set threshold for a second neuron to improve model accuracy
    :return: accuracy, AUROC, AUPRC
    """
    curr_test_loader = all_tasks_test_data[task_index]
    y = curr_test_loader.dataset.tensors[1].cuda()

    model.eval()
    with torch.no_grad():
        test_outputs = []

        for batch_X, batch_y, batch_mask in curr_test_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_mask = batch_mask.cuda()

            if use_MLP:
                outputs = model.forward(batch_X)
            else:
                outputs = model.forward(batch_X, batch_mask)

            test_outputs.append(outputs)

        acc, auroc, auprc = get_stats(test_outputs, y, acc_threshold)
        return acc * 100, auroc * 100, auprc * 100


def evaluate_current_task_PSP(model, all_tasks_test_data, task_index, use_MLP, contexts, layer_dimension):
    """
    Evaluate results on the first task, using the current PSP model.

    :param model: torch model instance
    :param all_tasks_test_data: list of all test dataloaders ([X_test, y_test, mask_test]) until the current task index
    :param task_index: index of the current task
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param contexts: binary context vectors (shape=(num_tasks, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size) and 1 (second dimension taken)
    :return: accuracy, AUROC, AUPRC
    """
    curr_test_loader = all_tasks_test_data[task_index]
    y = curr_test_loader.dataset.tensors[1].cuda()

    model.eval()
    with torch.no_grad():
        test_outputs = []

        for batch_X, batch_y, batch_mask in curr_test_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_mask = batch_mask.cuda()

            if use_MLP:
                outputs = model.forward(batch_X, True, contexts, task_index)
            else:
                outputs = model.forward(batch_X, batch_mask, True, contexts, task_index)

            test_outputs.append(outputs)

        acc, auroc, auprc = get_stats(test_outputs, y)
        return acc * 100, auroc * 100, auprc * 100


def evaluate_tasks_average(model, all_tasks_test_data, contexts, layer_dimension, superposition, task_index, use_MLP,
                           batch_size, adapter_trainable_layers=None, acc_thresholds=None, task_names_string=None):
    """
    Evaluate average results until the current task, using the current model.

    :param model: torch model instance
    :param all_tasks_test_data: list of all test data [X_test, y_test, mask_test] until the current task index
    :param contexts: binary context vectors (shape=(num_tasks-1, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size) and 1 (second dimension taken)
    :param superposition: boolean - True, if superposition is used
    :param task_index: index of the current task, which is being learned
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param batch_size: batch size
    :param adapter_trainable_layers: list of strings with the names of trainable adapter layers
    :param acc_thresholds: optional list of parameters to set threshold for a second neuron to improve model accuracy
    :param task_names_string: the order of learning tasks: 'NLP first', 'CV first', 'mixed', 'fixed NLP first',
                              'fixed CV first', 'fixed mixed' or 'Split CIFAR-100'
    :return: mean accuracy, mean AUROC, mean AUPRC (across tasks)
    """
    accs, aurocs, auprcs = [], [], []
    if superposition:
        for task_i in range(task_index, -1, -1):  # iterate across tasks backwards
            # evaluate results on the current task
            acc_threshold = acc_thresholds[task_i] if acc_thresholds is not None else None
            acc, auroc, auprc = evaluate_current_task(model, all_tasks_test_data, task_i, use_MLP, acc_threshold)
            accs.append(acc)
            aurocs.append(auroc)
            auprcs.append(auprc)

            # context multiplication to the previous task
            if task_i > 0:  # because we do not perform multiplication before the first task
                context_multiplication(model, contexts, layer_dimension, task_i - 1, adapter_trainable_layers)  # task_i - 1, because contexts are only used between tasks

        # restore model parameters to the old ones (before context multiplication)
        for task_i in range(task_index):  # iterate across tasks forward
            context_multiplication(model, contexts, layer_dimension, task_i, adapter_trainable_layers)
    else:
        for i in range(len(all_tasks_test_data)):
            acc, auroc, auprc = evaluate_current_task(model, all_tasks_test_data, i, use_MLP)
            accs.append(acc)
            aurocs.append(auroc)
            auprcs.append(auprc)

    # return np.mean(accs), np.mean(aurocs), np.mean(auprcs), []

    # # use the lines below (instead of the return above) if you want to know the average accuracy for NLP, CV and both, respectively
    # if superposition:
    #     accs.reverse()  # since accuracies are calculated from the last to the first task using superposition
    # if task_names_string == 'NLP first' or task_names_string == 'fixed NLP first':
    #     return np.mean(accs[:5]), np.mean(accs[5:]), np.mean(accs), []
    # elif task_names_string == 'CV first' or task_names_string == 'fixed CV first':
    #     return np.mean(accs[5:]), np.mean(accs[:5]), np.mean(accs), []
    # elif task_names_string == 'mixed' or task_names_string == 'fixed mixed':
    #     return np.mean(accs[::2]), np.mean(accs[1::2]), np.mean(accs), []

    # use the lines below (instead of the returns above) if you want to know the accuracies for all previous tasks, not just the mean
    if superposition:
        accs.reverse()  # since accuracies are calculated from the last to the first task
    return -1, -1, -1, accs


def evaluate_tasks_average_more_SuperAdapters(model, all_tasks_test_data, contexts, layer_dimension,
                                              task_index, use_MLP, change_superadapter, tasks_results,
                                              adapter_trainable_layers=None, acc_thresholds=None):
    """
    Evaluate average results until the current task, using the current model.

    :param model: torch model instance
    :param all_tasks_test_data: list of all test data [X_test, y_test, mask_test] until the current task index
    :param contexts: binary context vectors (shape=(num_tasks-1, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size) and 1 (second dimension taken)
    :param task_index: index of the current task, which is being learned
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :param change_superadapter: change SuperAdapter every change_superadapter-th task
    :param tasks_results: results for each task using more SuperAdapters, list of results per task, every task in the form of (acc, auroc, auprc)
    :param adapter_trainable_layers: list of strings with the names of trainable adapter layers
    :param acc_thresholds: optional list of parameters to set threshold for a second neuron to improve model accuracy
    :return: updated tasks_results, mean accuracy, mean AUROC, mean AUPRC (across tasks)
    """
    adapter_set_num = task_index // change_superadapter
    accs, aurocs, auprcs = [], [], []

    for task_i in range(task_index, (adapter_set_num * change_superadapter) - 1, -1):  # iterate across tasks backwards

        # # use saved biases from previous tasks
        # old_biases = []
        # if task_i != task_index:
        #     ind = 0
        #     for name, params in model.named_parameters():
        #         if name.endswith('bias') and "layer_norm" not in name and name in adapter_trainable_layers:
        #             with torch.no_grad():
        #                 old_biases.append(params)
        #                 bias = torch.load('models/bias_task%d_%d.pt' % (task_i, ind))
        #                 params.copy_(bias)
        #                 ind += 1

        # # use saved output layers from previous tasks
        # if task_i != task_index:
        #     for name, params in model.named_parameters():
        #         if name == "mlp.2.weight":
        #             with torch.no_grad():
        #                 old_output_layer = params
        #                 output_layer = torch.load('models/output_layer_task%d.pt' % task_i)
        #                 params.copy_(output_layer)


        # evaluate results on the current task
        acc_threshold = acc_thresholds[task_i] if acc_thresholds is not None else None
        acc, auroc, auprc = evaluate_current_task(model, all_tasks_test_data, task_i, use_MLP, acc_threshold)
        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)
        tasks_results[task_i] = (acc, auroc, auprc)

        # # restore old biases for the current task
        # if task_i != task_index:
        #     ind = 0
        #     for name, params in model.named_parameters():
        #         if name.endswith('bias') and "layer_norm" not in name and name in adapter_trainable_layers:
        #             with torch.no_grad():
        #                 params.copy_(old_biases[ind])
        #                 ind += 1

        # # restore old output layer for the current task
        # if task_i != task_index:
        #     for name, params in model.named_parameters():
        #         if name == "mlp.2.weight":
        #             with torch.no_grad():
        #                 params.copy_(old_output_layer)

        # context multiplication to the previous task
        if task_i % change_superadapter != 0:  # do not perform multiplication before the first task for the current set of adapters
            context_multiplication(model, contexts, layer_dimension, task_i - 1, adapter_trainable_layers)  # task_i - 1, because contexts are only used between tasks

    # restore model parameters to the old ones (before context multiplication)
    for task_i in range(adapter_set_num * change_superadapter, task_index, 1):  # iterate across tasks forward
        context_multiplication(model, contexts, layer_dimension, task_i, adapter_trainable_layers)

    return tasks_results, np.mean(accs), np.mean(aurocs), np.mean(auprcs)


def evaluate_tasks_average_PSP(model, all_tasks_test_data, contexts, layer_dimension, task_index, use_MLP):
    """
    Evaluate average results until the current task, using the current PSP model.

    :param model: torch model instance
    :param all_tasks_test_data: list of all test data [X_test, y_test, mask_test] until the current task index
    :param contexts: binary context vectors (shape=(num_tasks, num of model layers))
    :param layer_dimension: list of 0 (first dimension taken for context size) and 1 (second dimension taken)
    :param task_index: index of the current task, which is being learned
    :param use_MLP: boolean - if True use MLP, else use Transformer
    :return: mean accuracy, mean AUROC, mean AUPRC (across tasks)
    """
    accs, aurocs, auprcs = [], [], []

    for task_i in range(task_index, -1, -1):  # iterate across tasks backwards
        # evaluate results on the current task
        acc, auroc, auprc = evaluate_current_task_PSP(model, all_tasks_test_data, task_i, use_MLP, contexts, layer_dimension)
        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)

    return np.mean(accs), np.mean(aurocs), np.mean(auprcs)



