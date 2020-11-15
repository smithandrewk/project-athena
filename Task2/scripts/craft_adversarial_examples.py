#!/usr/bin/env python3
import sys
sys.path.append("../../src")
import argparse
import numpy as np
import os
import time
from matplotlib import pyplot as plt

from utils.model import load_lenet, load_pool
from utils.file import load_from_json
from utils.metrics import error_rate
from attacks.attack import generate
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def generate_ae(model, data, labels, attack_configs, eot=False, save=False, output_dir=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the adversarial examples.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")
    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    for id in range(num_attacks):
        print("STARTING NEW ATTACK") #AS
        key = "configs{}".format(id)
        attack_args = attack_configs.get(key)
        attack_args["eot"] = eot
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_args
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)
        description = str(attack_configs.get(key).get("description")) # AS

        # plotting some examples
        num_plotting = min(data.shape[0], 5) # change the second number to number of samples that you want to plot, otherwise, it will plot all of them
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap
            ='gray')
            title = '{}(EOT:{}): {}->{}'.format(description,
                                        "ON" if eot else "OFF",
                                        labels[i],
                                        predictions[i]
                                        )
            plt.title(title)
            # editor : Andrew Smith
            # description : saves individual adversarial figures
            initial_label = str(labels[i])#AS
            predicted_label = str(predictions[i])#AS
            adv_ex_dir = "../results/"+description
            if(not os.path.isdir(adv_ex_dir)):
                os.mkdir(adv_ex_dir)
            plt.savefig(adv_ex_dir+"/"+initial_label+"->"+predicted_label+".jpg")
            # end edit
            plt.show()
            plt.close()
        # save the adversarial example
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            # save with a random name
            file = os.path.join(output_dir, "{}.npy".format(description))
            print("Save the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-p', '--pool-configs', required=False,
                        default='../configs/athena-mnist.json')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/model-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/data-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='../configs/attack-zk-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print("------AUGMENT SUMMARY-------")
    print("POOL CONFIGS:", args.pool_configs)
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)
    attack_configs = load_from_json(args.attack_configs)
    pool_configs = load_from_json(args.pool_configs)

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=pool_configs,
                        model_configs=model_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble from the WD pool
    wds = list(pool.values())
    target = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    bs = np.load(data_file)

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)
    # option to subset samples and labels here
    number_of_samples = 5
    bs = bs[:number_of_samples]
    labels = labels[:number_of_samples]
    # Normal approach
    # Compute the loss w.r.t. a single input
    # For an ensemble target, averaging the losses of WDs'.
    generate_ae(model=target,
                data=bs, labels=labels,
                eot=False,
                save=True,
                output_dir=data_configs.get('dir'),
                attack_configs=attack_configs
                )

    # Adaptive approach (with EOT)
    # Compute the loss expectation over specific distribution.
    # For an ensemble target, averaging the EOT of WDs'.
    generate_ae(model=target,
                data=bs, labels=labels,
                eot=True,
                save=True,
                output_dir=data_configs.get('dir'),
                attack_configs=attack_configs
                )
