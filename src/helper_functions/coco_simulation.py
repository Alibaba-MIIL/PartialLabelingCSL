import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_class_freq(stat_orig, stat_sim):
    plt.figure()
    plt.plot(stat_orig, label="original")
    plt.plot(stat_sim, label="simulated")
    plt.xlabel("Class index")
    plt.ylabel("Class frequency")

    path_dest = "./outputs"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    plt.savefig(os.path.join(path_dest, "class_freq.png"))


def simulate_coco(args, dataset_train, mode="fix_per_class", param=1000):
    ''' Possible modes:
        random_per_sample (param: remove percentage, 0.1, 0.2,...)
        fix_per_class (param: number of pos/neg samples per class, 1000, 2000,...)
    '''

    # Parameters
    mode = args.simulate_partial_type
    param = args.simulate_partial_param
    save_class_frequencies = False

    targets_vec = dataset_train.targets_all
    S = np.array([y.numpy() for x, y in targets_vec.items()])
    # S = np.array([y.max(dim=0)[0].numpy() for x, y in targets_vec.items()])
    img_ids = list(dataset_train.targets_all.keys())

    # Original samples
    num_samples = S.sum(axis=0)
    stat_orig = num_samples / S.shape[0]
    print("Original stat:", stat_orig[:10])

    if mode == "fix_per_class" or mode == "fpc":
        print("Simulate coco. Mode: %s. Param: %f" % (mode, param))

        max_pos = int(param)
        max_neg = int(param)
        add_one_label = False
        Sout = -np.ones_like(S)
        for c in range(S.shape[1]):
            s = S[:, c]
            idx_pos = np.where(s == 1)[0]
            idx_neg = np.where(s == 0)[0]
            idx_select_pos = np.random.choice(idx_pos, np.minimum(max_pos, len(idx_pos)), replace=False)
            idx_select_neg = np.random.choice(idx_neg, np.minimum(max_neg, len(idx_neg)), replace=False)
            Sout[idx_select_pos, c] = 1
            Sout[idx_select_neg, c] = 0

        if add_one_label:
            # Add one positive label in case of no-positive labels found in sample (the same for negative)
            for i, x in enumerate(Sout):
                if not np.any(x == 1):
                    idx_pos = np.where(S[i] == 1)[0]
                    idx_select_pos = np.random.choice(idx_pos, 1)
                    Sout[i, idx_select_pos] = 1
                if not np.any(x == 0):
                    idx_neg = np.where(S[i] == 0)[0]
                    idx_select_neg = np.random.choice(idx_neg, 1)
                    Sout[i, idx_select_neg] = 0
        S = Sout

    elif mode == "random_per_sample" or mode == "rps":
        print("Simulate coco. Mode: %s. Param: %f" % (mode, param))

        idx_random = np.random.random((S.shape)) < param
        S[idx_random] = -1

    # Assign in sampler
    targets_all = dict(zip(img_ids, S))
    dataset_train.targets_all = targets_all

    # Simulated class frequencies
    num_samples = (S == 1).sum(axis=0)
    stat_simulate = num_samples / S.shape[0]
    print("Simulated stat:", stat_simulate[: 5])

    return dataset_train
