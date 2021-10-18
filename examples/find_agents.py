import os
from examples.Test_agent import evaluate_agent
import numpy as np
import json
import csv
import time

from rlpyt.utils.collections import namedarraytuple

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))


def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    return (paths)


if __name__ == "__main__":
    seed = 1234
    episodes = 100
    conf_paths = list_files('../data/local/', 'conf.json')
    first_initials = None
    mars = np.zeros(len(conf_paths))
    rewards = np.zeros((len(conf_paths),episodes))
    avg_time = np.zeros(len(conf_paths))
    start_time = time.time()
    for i, conf_path in enumerate(conf_paths):
        with open(conf_path, 'r+') as conf_file:
            config = json.loads(conf_file.read())
        conf_start_time = time.time()
        print(f'Running agent {i+1} of {len(conf_paths)}')
        mar, tot_rewards, initials = evaluate_agent(config, episodes, display=False, seed=seed)
        if first_initials is None:
            first_initials = initials
        assert np.allclose(first_initials, initials)
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
