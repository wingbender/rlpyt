
"""
Launches multiple experiment runs and organizes them on the local
compute resource.
Processor (CPU and GPU) affinities are all specified, to keep each
experiment on its own hardware without interference.  Can queue up more
experiments than fit on the machine, and they will run in order over time.  

To understand rules and settings for affinities, try using 
affinity = affinity.make_affinity(..)
OR
code = affinity.encode_affinity(..)
slot_code = affinity.prepend_run_slot(code, slot)
affinity = affinity.affinity_from_code(slot_code)
with many different inputs to encode, and see what comes out.

The results will be logged with a folder structure according to the
variant levels constructed here.

"""

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel


if __name__ == "__main__":
    # Either manually set the resources for the experiment:
    affinity_code = encode_affinity(
        n_cpu_core=4,
        n_gpu=0,
        # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
        # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
        cpu_per_run=2,
        set_affinity=True,  # it can help to restrict workers to individual CPUs
    )
    # Or try an automatic one, but results may vary:
    # affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

    runs_per_setting = 5
    # experiment_title = "compare"
    # experiment_title = "SACfD"
    experiment_title = "SAC"

    variant_levels = list()




    if experiment_title == "SAC":
        max_decorrelation = [0,5000,10000]
        values = list(zip(max_decorrelation))
        dir_names = [f"SAC_max_decor_{v[0]}" for v in values]
        keys = [("sampler", "max_decorrelation_steps")]
    if experiment_title == "SACfD":
        # Within a variant level, list each combination explicitly.
        expert_ratio = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        values = list(zip(expert_ratio))
        dir_names = [f"SACfD_ablation_ER_{v[0]}" for v in values]
        keys = [("algo", "expert_ratio")]
    if experiment_title == "compare":
        names = ["SAC","SACfD"]
        values = list(zip(names))
        dir_names = [f"Compare_{v[0]}" for v in values]
        keys = [("general", "algo")]
    variant_levels.append(VariantLevel(keys, values, dir_names))
    # batch_T = [1024, 1024]
    # # games = ["pong", "seaquest"]
    # values = list(zip(batch_T))
    # dir_names = [f"_T_{v}" for v in values]
    # keys = [('sampler','batch_T')]
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    # Between variant levels, make all combinations.
    variants, log_dirs = make_variants(*variant_levels)

    run_experiments(
        script="../examples/Single_Fly_SACfD.py",
        affinity_code=affinity_code,
        experiment_title=experiment_title,
        runs_per_setting=runs_per_setting,
        variants=variants,
        log_dirs=log_dirs,
    )