import json
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os


plt.rc('legend', fontsize=17)
plt.rc('axes', axisbelow=True)
params = {'text.latex.preamble': [r'\usepackage{amsfonts}',r'\usepackage[cm]{sfmath}'],
          'figure.autolayout': True, 'text.usetex': True,
          "svg.fonttype": 'none',
          'font.family': 'sans-serif',
          # 'font.sans-serif': ['Computer Modern'],
          'font.size':17,
          'legend.columnspacing' :0,
          'legend.handletextpad' :-0.1
        }
plt.rcParams.update(params)

### Configuration of the data ###
data_path = "datasets/results"
# data_path = "bert_performance/data"


# Datasets used to train the classifier
json_dataset_names = [fname for fname in os.listdir(data_path) if fname.startswith("with_pred")]
json_dataset_names.sort()

_training_dataset_names = ["DBLP/R", "M/M", "R/R"]
training_dataset_colors = ["C0", "C3", "C7"]
training_dataset_axis_cuts = ["1611", "5640", "16110"]
training_dataset_path_names = json_dataset_names


training_dataset_max_dict = dict(zip(_training_dataset_names,training_dataset_axis_cuts))
traing_dataset_colors_dict = dict(zip(_training_dataset_names,training_dataset_colors))
training_dataset_path_dict = dict(zip(_training_dataset_names,training_dataset_path_names))

# Datasets useed to test the classifier
_test_dataset_names = ["DBLP/R", "M/M", "W/R"]
test_dataset_colors = ["C0", "C3", "C2"]
test_dataset_ids = [1,0,2]

test_dataset_colors_dict = dict(zip(_test_dataset_names,test_dataset_colors))
test_dataset_ids_dict = dict(zip(test_dataset_ids,_test_dataset_names))


# Metrics to be displayed
metric_names = ["Test F1", "Test Recall","Test Prec."]#,"Test AUC","Test Accu."]
metric_markers =  ['v','o','X']#,'s','X']
metric_markers_dict = dict(zip(metric_names,metric_markers))


def metric_name2label(metric_name):
    return re.split('Test |Valid. ',metric_name)[1]

def svg2eps(filepath):
    pdf2psscript = "pdftops -eps " + filepath + ".pdf"
    os.system("bash -c '%s'" % pdf2psscript)

def load_json_performance(filepath):
    with open(filepath, 'r') as fp:
        _dataset_stats = json.load(fp)
    return _dataset_stats

def load_train_stats():
    json_dataset = {}
    for training_data_name, classifier_results_name in training_dataset_path_dict.items():
        json_dataset[training_data_name] = load_json_performance(os.path.join(data_path, classifier_results_name))

    prediction_dataset = {}

    for training_data_name, sample_count_dict in json_dataset.items():
        s = defaultdict(dict)
        for sample_count, seedList in sample_count_dict.items():
            for seedDict in seedList:
                _seed = list(seedDict.keys())[0]
                for seed, stats_dict in seedDict.items():
                    train_stats = stats_dict['train_stats']
                    p = {}
                    for e in train_stats:
                        epoch = e['epoch']
                        p[epoch] = e
                    s[sample_count][seed] = p
        prediction_dataset[training_data_name] = s
    return prediction_dataset


def get_best_epoch_by_min_val_loss(train_stats, modelname):
    max_sample_key = str(max([ int(sample_count) for sample_count in list(train_stats[modelname].keys())]))
    best_epoch_dict = {}
    for seed, seed_dict in train_stats[modelname][max_sample_key].items():
        val_loss_dict = {}
        for epoch, d in seed_dict.items():
            val_loss_dict[epoch] = d['Valid. Loss']
        best_epoch_dict[seed] = min(val_loss_dict,key=val_loss_dict.get)
    return best_epoch_dict

def best_model_epoch_dict():
    best_model_epoch_dict = {}
    train_stats = load_train_stats()
    for model_name in _training_dataset_names:
        best_model_epoch_dict[model_name] = get_best_epoch_by_min_val_loss(train_stats, model_name)
    return  best_model_epoch_dict

def best_epoch_for_seed(seed):
    d = best_model_epoch_dict()
    m = {}
    for model_name, seed_dict in d.items():
        m[model_name] = seed_dict[seed]
    return m

def load_prediction_dataset():
    json_dataset = {}
    for training_data_name, classifier_results_name in training_dataset_path_dict.items():
        json_dataset[training_data_name] = load_json_performance(os.path.join(data_path, classifier_results_name))

    prediction_dataset = {}

    for training_data_name, sample_count_dict in json_dataset.items():
        s = defaultdict(dict)
        for sample_count, seedList in sample_count_dict.items():
            for seedDict in seedList:
                _seed = list(seedDict.keys())[0]
                for seed, stats_dict in seedDict.items():
                    predict_stats = stats_dict['predict_stats']
                    p = defaultdict(dict)
                    for e in predict_stats:
                        test_dataset_name = test_dataset_ids_dict[e['TestSet']]
                        epoch = e['epoch']
                        p[test_dataset_name][epoch] = e
                    s[sample_count][seed] = p
        prediction_dataset[training_data_name] = s
    return prediction_dataset

def get_best_epoch(epoch_dict,by_metric = "Test F1"):
    metric_scores = []
    for epoch, p in epoch_dict.items():
        metric_scores.append(p[by_metric])
    best_epoch = metric_scores.index(max(metric_scores))
    return best_epoch + 1

def _box_plot_data(prediction_dataset,training_dataset_name, test_dataset_name, metric):
    sampleNoList, y_data = [], []
    for sample_count, exp_dict in prediction_dataset[training_dataset_name].items():
        y_series = []
        for exp_seed, test_dataset_dict in exp_dict.items():
            best_epoch = get_best_epoch(test_dataset_dict[test_dataset_name])
            score = test_dataset_dict[test_dataset_name][best_epoch][metric]
            y_series.append(score)
        sampleNoList.append(int(sample_count))
        y_data.append(y_series)
    return sampleNoList, y_data

def recall_complementary_binary(flat_true_labels, flat_pred_labels_a, flat_pred_labels_b):
    key_set = set(flat_true_labels)
    k = np.array(flat_true_labels)
    a = np.array(flat_pred_labels_a)
    b = np.array(flat_pred_labels_b)
    recall_comp = {}
    for key in key_set:
        _a_wrong = a[k == key] != key
        _b_wrong = b[k == key] != key
        intersection_ab = [ 1 for e1,e2 in zip(_a_wrong,_b_wrong) if (e1 and e2)]
        intersection_ab_count = np.sum(intersection_ab)
        _a_wronge_count = np.sum(_a_wrong)
        recall_comp[key] = intersection_ab_count/_a_wronge_count
    return recall_comp

def main_doulbe_braxes(metric_name = "Test F1", test_series_name_output = "W/R", train_series_name_output = "R/R"):
    output_series = {}

    from matplotlib.gridspec import GridSpec
    prediction_dataset = load_prediction_dataset()
    r1 = 10
    r2 = 1
    r3 = 1

    ratios = [r1, r2, r3]

    start1 = 200
    end1 = 1600

    start2 = 5640
    end2 = start2 + 100 #+ (end1 - start1) * r2 / r1

    start3 = 16110
    end3 = start3 + 100 #(end1 - start1) * r3 / r1

    fig_bp, ax_bp = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    gs = GridSpec(1, 3, figure=fig_bp, width_ratios=ratios)
    gs.update(wspace=0.025, hspace=0.05)
    axes = [plt.subplot(g) for g in gs]
    # fig_bp, axes = plt.subplots( nrows=1, ncols=3, figsize=(10, 6), sharex=True, sharey=False, gridspec_kw={'width_ratios':ratios})
    # axes = ax_bp
    startends = [(start1, end1), (start2, end2), (start3, end3)]
    # fig, _axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    text_base_y = 0.3

    ##### Set the series to be plotted
    # test_dataset_names = _test_dataset_names[0:2]
    test_dataset_names = _test_dataset_names
    training_dataset_names = _training_dataset_names

    N_test = len(test_dataset_names)
    N_train = len(training_dataset_names)

    w4 = 200
    eps2 = 5
    w3 = w4-2*eps2
    w2 = w3/N_test
    eps1 = 2
    w1 = w2-2*(eps1)
    w0 = w1/(N_train)

    test_dataset_shift = {}
    for i, test_dataset_name in enumerate(test_dataset_names):
        test_dataset_shift[test_dataset_name] = -w4/2 + eps2 + i*w2

    training_dataset_shift = {}

    for i, training_data_name in enumerate(training_dataset_names):
        training_dataset_shift[training_data_name] = eps1 + (i+.5)*w0


    for n, (ax, startend, r) in enumerate(zip(axes, startends, ratios)):
        for test_dataset_name in test_dataset_names:
            x_ranges = []
            for test_pos in range(startend[0],startend[1]+w4,w4):
                x_ranges.append((test_pos + test_dataset_shift[test_dataset_name],w2))

            ax.broken_barh(x_ranges, (0, 1), facecolor=test_dataset_colors_dict[test_dataset_name], alpha=0.2)

            bps = []
            for training_data_name in training_dataset_names:
                sampleNoList, y_data = _box_plot_data(prediction_dataset, training_data_name, test_dataset_name, metric_name)
                training_pos = np.array(sampleNoList) + test_dataset_shift[test_dataset_name] + training_dataset_shift[training_data_name]
                bp = ax.boxplot(y_data, sym='', positions=training_pos, manage_ticks=False,
                                widths=w0, patch_artist=True,
                                flierprops=dict(color=traing_dataset_colors_dict[training_data_name],
                                                facecolor=traing_dataset_colors_dict[training_data_name]),
                                boxprops=dict(alpha=1, facecolor=traing_dataset_colors_dict[training_data_name],
                                              edgecolor=traing_dataset_colors_dict[training_data_name]),
                                medianprops=dict(color='k')
                                )
                bps.append(bp)
                if (test_series_name_output == test_dataset_name) and (train_series_name_output == training_data_name):
                    for sNo, y in zip(sampleNoList, y_data):
                        output_series[str(sNo)] = y

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(bottom=text_base_y, top=1.01)
        if n != 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft='off')
            ax.tick_params(axis='y',
                            # which='both',
                            left=False,  # ticks along the top edge are off
                            labelleft=False, labelright=False)

            ax.set_xticks([startend[0]])

        d = .015
        angle = np.pi / 2 - 0.1
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

        if n == 0:
            ax.plot(np.array([1, 1]) + np.cos(angle) * np.array((-d, d)), np.sin(angle) * np.array((-d, +d)), **kwargs)
            ylabel = metric_name.replace("Test ", "")
            ax.set_xlabel('Samples size')
            ax.set_ylabel(ylabel)
            ax.set_xlim(startend[0] - 100, startend[1] + 100)

            ### legend
            # ---- DBLP/R ----
            d_training, = ax.plot([], [], c=traing_dataset_colors_dict['DBLP/R'], fillstyle='left', markersize=20, marker='s', linestyle='none')
            d_test, = ax.plot([], [], c=test_dataset_colors_dict['DBLP/R'], fillstyle='right', markersize=20, alpha=0.2, marker='s', linestyle='none')
            # ---- M/M ----
            m_training, = ax.plot([], [], c=traing_dataset_colors_dict['M/M'], fillstyle='left', markersize=20, marker='s', linestyle='none')
            m_test, = ax.plot([], [], c=test_dataset_colors_dict['M/M'], fillstyle='right', markersize=20, alpha=0.2, marker='s', linestyle='none')
            # ---- R/R ----
            r_training, = ax.plot([], [], c=traing_dataset_colors_dict['R/R'],marker='s', markersize=20, linestyle='none' )
            # ---- W/R ----
            w_testing, = ax.plot([], [], c=test_dataset_colors_dict['W/R'], marker='s', markersize=20, alpha=0.2, linestyle='none')
            # ---- Plot Legend ----
            ax.legend(((d_training, d_test), (m_training, m_test), r_training, w_testing),
                      ('DBLP', 'Manual', 'RegEx', 'Wikidata'), loc="lower right")
            ###

        if n == 1:
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.set_xlim(startend[0] - 100, startend[0] + 100)
        if n == 2:
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.set_xlim(startend[0] - 100, startend[0] + 100)
        ax.yaxis.grid()

    fig_name = "figures/sampleNo_vs_" + ylabel + "_score_on_" + "both" + ".svg"
    plt.tight_layout()
    fig_bp.savefig(fig_name, transparent=True)
    plt.show()

    return prediction_dataset, fig_bp, ax_bp, output_series


for metric_name in metric_names:
    result_series, fig_bp, ax_bp, output_series = main_doulbe_braxes(metric_name)
