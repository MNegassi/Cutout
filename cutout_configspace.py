import json
import sys
import os
import argparse

import ConfigSpace as CS
from os.path import abspath, join as path_join

import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
from hpbandster.config_generators.kde_ei import KDEEI

from train import run_cutout

def get_config_space(seed=None):
    # XXX: Change lower upper and default
    cs = CS.ConfigurationSpace(seed)
    HPs = [
        CS.UniformIntegerHyperparameter("length", lower=1, upper=64, default_value=2),
        CS.UniformIntegerHyperparameter("n_holes", lower=1, upper=32, default_value=16),
    ]

    [cs.add_hyperparameter(hp) for hp in HPs]

    return cs


def main():
    # XXX: Insert hyperband stuff here

    parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')



    parser.add_argument(
        "--model", help="Neural network to be trained with augmented data",
        default="resnet18"
    )

    parser.add_argument(
        "--dataset", help="Dataset to train neural network on",
        default="cifar10"
    )

    parser.add_argument(
        "--max_epochs", default=160, help="Maximum number of epochs to train network", type=int
    )

    parser.add_argument(
        "--optimizer", default="hyperband", help="Number of successive halving for hyperband",
    )

    parser.add_argument(
        "--run-id", help="The id of single job"
    )

    parser.add_argument("--seed", help="Random seed.", default=1)

    args = parser.parse_args()

    config_space = get_config_space(seed=args.seed)

    # this run hyperband sequentially

    class CutoutWorker(Worker):
        def __init__(self, model="resnet18", dataset="cifar10", *args, **kwargs):
            self.model = model
            self.dataset = dataset
            super().__init__(*args, **kwargs)

        def compute(self, config, budget, *args, **kwargs):
            """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these kinds of failures.
            """

            results= run_cutout(
                epochs=int(budget), model=self.model, dataset=self.dataset,
                cutout=True, data_augmentation=True,
                n_holes=config["n_holes"], length=config["length"]
            )
            test_error = results["test_error"]

            return({
                'loss': test_error,   # this is the a mandatory field to run hyperband
                'info': results  # can be used for any user-defined information - also mandatory
            })


    # starts a local nameserve
    from hpbandster.distributed import utils as distributed_utils
    nameserver, ns_port = distributed_utils.start_local_nameserver()


    # import the definition of the worker (could be in here as well, but is imported to reduce code duplication)

    # starting the worker in a separate thread
    w = CutoutWorker(
        dataset=args.dataset, model=args.model,
        nameserver=nameserver, ns_port=ns_port,
    )
    w.run(background=True)

    # simple config space here: just one float between 0 and 1

    if args.optimizer == "BOHB":
        print("Using Model Based Hyperband")
        CG = KDEEI(config_space, mode="sampling", num_samples=64)  # model-based hyperband
    else:
        print("Using Hyperband")
        CG = hpbandster.config_generators.RandomSampling(config_space)  # hyperband on steriods


    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=CG,
        run_id='0',
        eta=2,
        min_budget=20,
        max_budget=args.max_epochs,
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=(0, 1)
    )
    # runs one iteration if at least one worker is available, first parameter
    # is number of successive halving
    res = HB.run(5, min_n_workers=1)
    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)

    # Save results
    path = path_join(abspath("."), "AutoData/{}/cutout".format(args.dataset))

    # Get important information about best configuration from HB result object
    best_config_id = res.get_incumbent_id()  # Config_id of the incumbent with smallest loss
    best_run = res.get_runs_by_id(best_config_id)[-1]
    best_config_trajectory = res.get_incumbent_trajectory()


    json_data = {
        "best_config_id": best_config_id,
        "best_run_info": best_run.info,
        "best_config_trajectory": best_config_trajectory
    }

    # XXX: change output path to be in the arguements

    with open(os.path.join(path, "{}_{}_{}.json".format(args.optimizer, args.dataset, args.run_id)), "w") as fh:
        json.dump(json_data, fh)


if __name__ == "__main__":
    main()
