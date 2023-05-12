import math
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data, x_label, y_label, bins):
    """
    Plot histogram.

    :param data: numbers in a list
    :param x_label: label on x axis
    :param y_label: label on y axis
    :param bins: number of bins in a histogram
    :return: None
    """
    plt.hist(data, density=False, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_multiple_histograms(data, num_tasks, metrics, title, colors, y_label, y_min):
    """
    Plot multiple vertical bars for each task.

    :param data: dictionary with mean and std data for each task
    :param num_tasks: number of tasks trained
    :param metrics: list of strings - possibilities: 'acc', 'auroc', 'auprc'
    :param title: plot title (string)
    :param colors: list of colors used for bars (len(colors)=len(metrics))
    :param y_label: label of axis y (string)
    :param y_min: minimum y value
    :return: None
    """
    metrics_names = {
        'acc': 'Accuracy',
        'auroc': 'AUROC',
        'auprc': 'AUPRC'
    }

    # # if you want separate average accuracies for NLP tasks only, CV tasks only and both, uncomment the lines below
    # metrics_names = {
    #     'acc': 'NLP only average',
    #     'auroc': 'CV only average',
    #     'auprc': 'Total average'
    # }

    font = {'size': 20}
    plt.rc('font', **font)
    plt.grid(axis='y')

    bar_width = 0.5
    for i, m in enumerate(metrics):
        heights = [data[i][m] for i in range(num_tasks)]
        x_pos = [(n * len(metrics)) + (i * bar_width) for n in range(num_tasks)]
        plt.bar(x_pos, heights, width=bar_width, color=colors[i], edgecolor='black',
                yerr=[data[i]['std_' + m] for i in range(num_tasks)], capsize=7, label=metrics_names[m])

        # # plot numbers of mean (height) on every bar
        # for j, x in enumerate(x_pos):
        #     plt.text(x - 0.2, y_min + 1, round(heights[j], 1), {'size': 10})

    ax = plt.gca()
    ax.set_ylim([y_min, 100])

    plt.xticks([(i * len(metrics)) + (math.floor(len(metrics) / 2) * bar_width) for i in range(num_tasks)],
               ['Task 1' if i == 0 else 'Task 1-%d' % (i+1) for i in range(num_tasks)])

    plt.ylabel(y_label)
    # plt.title(title)
    plt.legend()
    plt.show()


def plot_multiple_results(num_tasks, num_epochs, first_average, means, stds, legend_lst, title, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param num_tasks: number of tasks trained
    :param num_epochs: number of epochs per task
    :param first_average: string - show results on 'first' task only or the 'average' results until current task index
    :param means: list - [mean_acc, mean_auroc, mean_auprc]
    :param stds: list - [std_acc, std_auroc, std_auprc]
    :param legend_lst: list of label values (len(legend_lst)=len(means))
    :param title: plot title (string)
    :param colors: list of colors used for lines (len(colors)=len(means))
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)
    plt.grid(axis='y')

    if num_tasks * num_epochs == len(means[0]):
        # plot horizontal lines to explain learning
        for i in range(num_tasks):
            x_min = i * num_epochs if i == 0 else (i * num_epochs) - 1
            x_max = ((i + 1) * num_epochs) - 1
            plt.hlines(y=103, xmin=x_min, xmax=x_max)
            plt.vlines(x=x_min, ymin=102, ymax=104)
            plt.vlines(x=x_max, ymin=102, ymax=104)

            plt.text(x=x_min + 1, y=104, fontsize=12,
                     s='Learning %s; Results for %s' % (str(i+1), str(1) if first_average == 'first' else '1-%s' % (i+1) if i != 0 else str(1)))

    # plot lines with confidence intervals
    i = 0
    for mean, std in zip(means, stds):
        # plot the shaded range of the confidence intervals (mean +/- std)
        if show_CI:
            up_limit = mean + std
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - std
            plt.fill_between(range(len(std)), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top
        plt.plot(range(len(mean)), mean, colors[i], linewidth=3)

        i += 1

    if legend_lst:
        plt.legend(legend_lst, loc='lower left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()


def plot_task_results(data, method_names, markers, colors, x_ticks, x_label, y_label):
    """
    Plot results for different methods.

    :param data: 2D list of results with the shape (num_methods, num_tasks)
    :param method_names: list of methods' names (len=num_methods)
    :param markers: list of point markers (len=num_methods)
    :param colors: list of colors (len=num_methods)
    :param x_ticks: list of strings to show at x axis (len=num_tasks)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :return: None
    """
    font = {'size': 25}
    plt.rc('font', **font)
    plt.grid(axis='y')

    # plt.yticks([75, 80, 85, 90, 95])

    x = list(range(1, 7))

    for i, method_res in enumerate(data):
        plt.plot(x, method_res, label=method_names[i], linestyle='--', color=colors[i], marker=markers[i],
                 linewidth=2, markersize=12)

    plt.xticks(x, x_ticks)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_heatmap(data, full_value):
    """
    Plot heatmap for the ablation study.

    :param data: 2D list of shape (num_layers, num_layers)
    :param full_value: value of the result without ablation
    :return: None
    """
    import seaborn as sns
    import numpy as np

    font = {'size': 25}
    plt.rc('font', **font)

    data = np.array(data)
    data[data == 0] = full_value

    relative_diff = (-1 + (data / full_value)) * 100

    ax = sns.heatmap(relative_diff, cmap=sns.color_palette("Blues_r", as_cmap=True), linecolor='black', linewidth=1, square=True)

    for i in range(len(data)):
        for j in range(len(data[i])):
            if relative_diff[i, j] < 0:   # upper triangle
                text = ax.text(j + 0.5, i + 0.5, round(relative_diff[i, j], 1), ha="center", va="center", color="black")

    ax.set_xticklabels(list(range(1, 7)))
    ax.set_yticklabels(list(range(1, 7)))

    plt.xlabel('last ablated layer')
    plt.ylabel('first ablated layer')
    plt.show()


def plot_accuracies_all_tasks(data, num_tasks, title, task_names):
    """
    Plot accuracies of all previous tasks like vertical bars.

    :param data: (all_tasks_accuracies_mean, all_tasks_accuracies_std), both are (num_tasks x num_tasks) lower triangular matrices
    :param num_tasks: number of tasks trained
    :param title: plot title (string)
    :param task_names: list of tasks' names as strings
    :return: None

    """
    font = {'size': 15}
    plt.rc('font', **font)

    # dict of precalculated tasks' upper bound accuracies if trained from scratch (from randomly initialized network)
    upper_bound_accuracies_std = {
        'HS': [87.5, 0.6],
        'SA': [72.9, 0.4],
        'S': [99.0, 0.5],
        'SA_2': [68.5, 1.6],
        'C': [97.0, 0.2],
        'CIF1': [41.7, 2.4],
        'CIF2': [40.1, 3.0],
        'CIF3': [46.5, 1.2],
        'CIF4': [43.3, 1.6],
        'CIF5': [48.7, 1.8]
    }

    # create 10 sections with varying number of bars
    sections = []
    for i in range(1, num_tasks + 1):
        section = [data[0][i - 1, j] for j in range(i)]
        section.append(upper_bound_accuracies_std[task_names[i-1]][0])
        sections.append(section)

    # create a figure and axis object
    fig, ax = plt.subplots()

    # set the x and y limits of the plot
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 100)

    # define a list of colors to use for the bars
    colors = {
        'HS': '#1f77b4',
        'SA': '#ff7f0e',
        'S': '#2ca02c',
        'SA_2': '#d62728',
        'C': '#9467bd',
        'CIF1': '#8c564b',
        'CIF2': '#e377c2',
        'CIF3': '#7f7f7f',
        'CIF4': '#bcbd22',
        'CIF5': '#17becf',
        'CIF6': '#1f77b4',
        'CIF7': '#ff7f0e',
        'CIF8': '#2ca02c',
        'CIF9': '#d62728',
        'CIF10': '#9467bd'
    }

    # plot each section as a set of bars with unique colors
    x_pos = 1
    tick_positions = []
    tick_labels = []
    for i, section in enumerate(sections):
        bar_num = 1
        x_values = [x_pos + j for j in range(len(section))]
        y_values = section
        for j in range(len(section)):
            if j == len(section) - 1:   # last bar - upper bound
                ax.bar(x_values[j], y_values[j], yerr=upper_bound_accuracies_std[task_names[i]][1], capsize=3, color='0.25')
                # ax.text(x_values[j], 2, 'ub', ha='center', fontsize=8, color='white')
                ax.text(x_values[j], 2, 'sn', ha='center', fontsize=8, color='white')
            else:
                ax.bar(x_values[j], y_values[j], yerr=data[1][i, j], capsize=3, color=colors[task_names[j]])
                ax.text(x_values[j], 2, f'{bar_num}', ha='center', fontsize=10)
                bar_num += 1
        tick_positions.append(np.mean(x_values))
        if i == 0:
            tick_labels.append('Task 1')
        else:
            tick_labels.append(f'Task 1-{i + 1}')
        x_pos += len(section) + 1

    # add labels and title
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)

    # set the xticks to be section names instead of numbers
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)

    # show the plot
    plt.show()


def plot_superposition_capacity(superposition_acc_mean, separate_networks_acc_mean,
                                superposition_acc_std, separate_networks_acc_std, title):
    """
    Plot how superposition capacity changes with enlarging the network
    Comparison of 10 separate networks and 1 superposition network accuracies.

    :param superposition_acc_mean: list of mean accuracies for 1 superimposed network with N neurons in the hidden layer
    :param separate_networks_acc_mean: list of mean accuracies for 10 separate networks with N/10 neurons in the hidden layer
    :param superposition_acc_std: list of accuracies' standard deviation for 1 superimposed network with N neurons in the hidden layer
    :param separate_networks_acc_std: list of accuracies' standard deviation for 10 separate networks with N/10 neurons in the hidden layer
    :param title: title of the plot as a string (dataset name)
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)

    neurons = np.array([10, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000])
    x = np.array(list(range(len(neurons))))

    bar_width = 0.3
    opacity = 0.6

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - (bar_width / 2), superposition_acc_mean, bar_width,
                    yerr=superposition_acc_std, capsize=5,
                    alpha=opacity,
                    color='b',
                    label='1 superimposed network')

    rects2 = ax.bar(x + (bar_width / 2), separate_networks_acc_mean, bar_width,
                    yerr=separate_networks_acc_std, capsize=5,
                    alpha=opacity,
                    color='r',
                    label='10 separate networks')

    ax.set_xlabel('# neurons in superimposed hidden layer')
    ax.set_ylabel('Accuracy (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(neurons)
    ax.legend()

    plt.title('dataset: ' + title)
    plt.show()


def plot_magnifying_factors(neurons, magnifying_factors):
    """
    Plot magnifying factors for 10 separate networks to reach one superposition network capacity.

    :param neurons: a list of the number of neurons in the superimposed hidden layer
    :param magnifying_factors: 2d list of magnifying factors, first column is for Split CIFAR 100 and
                               the second column is for 10 mixed NLP and CV tasks
    :return: None
    """
    font = {'size': 20}
    plt.rc('font', **font)

    x = neurons
    y1 = [val[0] for val in magnifying_factors]
    y2 = [val[1] for val in magnifying_factors]

    bar_width = 10
    opacity = 0.8

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - (bar_width / 2), y1, bar_width,
                    alpha=opacity,
                    color='b',
                    label='Split CIFAR-100')

    rects2 = ax.bar(x + (bar_width / 2), y2, bar_width,
                    alpha=opacity,
                    color='g',
                    label='10 mixed NLP and CV tasks')

    ax.set_xlabel('# neurons in superimposed hidden layer')
    ax.set_ylabel('magnification factor to achieve superposition capacity')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    pass

    # plot_magnifying_factors(np.array([10, 50, 100, 200, 300]), [[7, 2], [3, 2.8], [2.4, 2.5], [1.50, 1.35], [1.03, 1.07]])

    # plot_superposition_capacity(np.array([19.2, 33.4, 38.5, 40.3, 40.8, 42.2, 41.5, 42.7, 42.1, 40.9]),
    #                             np.array([10.4, 14.9, 24.8, 36.9, 40.0, 43.5, 43.9, 45.1, 45.5, 46.3]),
    #                             np.array([1.6, 2.1, 1.6, 1.1, 1.1, 0.9, 0.7, 1.6, 1.1, 1.3]),
    #                             np.array([0.5, 0.9, 1.7, 2.5, 0.4, 0.6, 0.3, 0.2, 1.2, 1.0]), 'Split CIFAR-100')

    plot_superposition_capacity(np.array([40.5, 54.9, 59.3, 59.7, 60.8, 60.3, 61.6, 61.7, 60.4, 59.7]),
                                np.array([29.9, 47.7, 51.8, 57.0, 60.7, 63.3, 63.6, 63.6, 64.6, 64.4]),
                                np.array([2.2, 1.9, 1.2, 1.2, 1.3, 1.2, 0.8, 0.4, 1.0, 1.0]),
                                np.array([8.3, 3.1, 1.1, 3.1, 0.7, 1.0, 0.9, 0.7, 0.4, 0.8]), '10 mixed NLP and CV tasks')


