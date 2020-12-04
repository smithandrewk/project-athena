#!/usr/bin/env python3
import sys
sys.path.append("../../src")
import argparse
import numpy as np
import os
import time
from matplotlib import pyplot as plt, image

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY
from models.image_processor import transform

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.
    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.
    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.
        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }
    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)
    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.
    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.
    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def evaluate(trans_configs, model_configs,
             data_configs, save=False, output_dir=None):
    """
    Evaluate Weak Defense
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # Load the undefended model (UM)
    file = os.path.join(model_configs.get('dir'), model_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    ensemble = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)[:500]
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)[:500]

    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')
    UM = []
    e = []
    PGDADT = []
    for i in range(len(ae_list)):
        ae_file = os.path.join(data_configs.get('dir'), ae_list[i])
        x_adv = np.load(ae_file)
        print(ae_file)
        # evaluate the undefended model on the AE
        print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
        pred_adv_um = undefended.predict(x_adv)
        err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['UM'] = err_um
        UM.append(err_um)

        # evaluate the ensemble on the AE
        print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
        pred_adv_ens = ensemble.predict(x_adv)
        err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['Ensemble'] = err_ens
        e.append(err_ens)

        # evaluate the baseline on the AE
        print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
        pred_adv_bl = baseline.predict(x_adv)
        err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['PGD-ADT'] = err_bl
        PGDADT.append((err_bl))

        print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))
    FGSM_rotation = [ae_list[0],ae_list[1],ae_list[2]]
    FGSM_rotation_UM = [UM[0],UM[1],UM[2]]
    FGSM_rotation_e = [e[0],e[1],e[2]]
    FGSM_rotation_PGDADT = [PGDADT[0],PGDADT[1],PGDADT[2]]
    BIM_rotation = [ae_list[3],ae_list[4],ae_list[5]]
    BIM_rotation_UM = [UM[3],UM[4],UM[5]]
    BIM_rotation_e = [e[3],e[4],e[5]]
    BIM_rotation_PGDADT = [PGDADT[3],PGDADT[4],PGDADT[5]]
    PGD_rotation = [ae_list[6],ae_list[7],ae_list[8]]
    PGD_rotation_UM = [UM[6],UM[7],UM[8]]
    PGD_rotation_e = [e[6],e[7],e[8]]
    PGD_rotation_PGDADT = [PGDADT[6],PGDADT[7],PGDADT[8]]
    ##
    data = {
        "UM": FGSM_rotation_UM,
        "Ensemble": FGSM_rotation_e,
        "PGD-ADT": FGSM_rotation_PGDADT
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.xticks(range(3),FGSM_rotation)
    plt.ylabel("Error")
    plt.gcf().set_size_inches(15, 10)
    plt.savefig('FGSM_rotation.jpg',dpi=100,bbox_inches='tight')
    plt.show()
    ##
    data = {
        "UM": BIM_rotation_UM,
        "Ensemble": BIM_rotation_e,
        "PGD-ADT": BIM_rotation_PGDADT
    }
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.xticks(range(3),BIM_rotation)
    plt.ylabel("Error")
    plt.gcf().set_size_inches(15, 10)
    plt.savefig('BIM_rotation.jpg',dpi=100,bbox_inches='tight')
    plt.show()
    ##3
    data = {
        "UM": PGD_rotation_UM,
        "Ensemble": PGD_rotation_e,
        "PGD-ADT": PGD_rotation_PGDADT
    }
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.xticks(range(3),PGD_rotation)
    plt.ylabel("Error")
    plt.gcf().set_size_inches(15, 10)
    plt.savefig('PGD_rotation.jpg',dpi=100,bbox_inches='tight')
    plt.show()
    ##

    # plt.bar(ae_list,UM)
    # plt.title("Evaluation against undefended model")
    # plt.ylabel("Error")
    # plt.xticks(rotation=45,fontsize=5)
    # plt.show()

    # plt.bar(ae_list,e)
    # plt.title("Evaluation against vanilla athena model")
    # plt.ylabel("Error")
    # plt.xticks(rotation=45,fontsize=5)
    # plt.show()

    # plt.bar(ae_list,PGDADT)
    # plt.title("Evaluation against PGD-ADT model")
    # plt.ylabel("Error")
    # plt.xticks(rotation=45,fontsize=5)
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/model-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/data-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print('------AUGMENT SUMMARY-------')
    print('TRANSFORMATION CONFIGS:', args.trans_configs)
    print('MODEL CONFIGS:', args.model_configs)
    print('DATA CONFIGS:', args.data_configs)
    print('OUTPUT ROOT:', args.output_root)
    print('DEBUGGING MODE:', args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    trans_configs = load_from_json(args.trans_configs)
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)

    # -------- test transformations -------------
    evaluate(trans_configs=trans_configs,
             model_configs=model_configs,
             data_configs=data_configs,
             save=True,
             output_dir=args.output_root)
