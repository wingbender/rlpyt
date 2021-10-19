import os

import pandas as pd

from examples.Test_agent import evaluate_agent
import numpy as np
import json
import csv
import time
import pandas as pd
import multiprocessing as mp

from rlpyt.utils.collections import namedarraytuple

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

def agregate_progress(base_path):
    paths = list_files(base_path,'progress.csv')
    avg_ret = {}
    # max_l = 0
    for i,path in enumerate(paths):
        try:
            tmp = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f'Couldn\'t read file {path}, no data')
            continue
        if 'Return/Average' in tmp.columns:
            avg_ret[path] = tmp['Return/Average']
        elif 'ReturnAverage' in tmp.columns:
            avg_ret[path] = tmp['ReturnAverage']
    df = pd.DataFrame.from_dict(avg_ret)
    df.to_csv('all_progress.csv')


def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    return (paths)


if __name__ == "__main__":
    agregate_progress('../data/')
    if False:
        multi = True
        seed = 1234
        episodes = 100
        conf_paths = list_files('../data/local/', 'conf.json')
        first_initials = None
        mars = np.zeros(len(conf_paths))
        rewards = np.zeros((len(conf_paths),episodes))
        avg_time = np.zeros(len(conf_paths))
        start_time = time.time()
        if multi:
            with mp.Manager() as mgr:
                print(p.map(f, [1, 2, 3]))
        else:
            for i, conf_path in enumerate(conf_paths):
                with open(conf_path, 'r+') as conf_file:
                    config = json.loads(conf_file.read())
                conf_start_time = time.time()
                print(f'Running agent {i+1} of {len(conf_paths)}')
                mar, tot_rewards, initials = evaluate_agent(config, episodes, display=False, seed=seed)
                # if first_initials is None:
                #     first_initials = initials
                # assert np.allclose(first_initials, initials)
                mars[i] = mar
                rewards[i,:] = tot_rewards
                now = time.time()
                avg_time[i] = (now - conf_start_time) / episodes
                avg_time_overall = (now - start_time) / (episodes * (i + 1))
                print(f"{conf_path}\t{mar}")
                print(f"avg episode time for agent:{avg_time[i]}")
                print(f"avg episode time overall:{avg_time_overall}")
                with open('../data/conf_mars.csv', 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(list(zip(conf_paths, mars.tolist(),avg_time.tolist())))
                np.savetxt('../data/rewards.csv',rewards)
