from __future__ import print_function, division
import os
import json
import h5py

from utils import flag_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate

from tensorboardX import SummaryWriter
from torch.multiprocessing import Manager

os.environ["OMP_NUM_THREADS"] = "1"

def hdf5_to_dict(hdf5_file_path):
    data = {}
    manager = Manager()
    md_data = manager.dict()
    with h5py.File(hdf5_file_path, 'r') as read_file:
        for scene in tqdm(read_file.keys()):
            # data[scene] = {}
            data[scene] = manager.dict()
            for pos in read_file[scene].keys():
                data[scene][pos] = read_file[scene][pos][()]
    md_data.update(data)
    return md_data

def main():
    args = flag_parser.parse_arguments()

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    args.episode_type = "TestValEpisode"
    args.test_or_val = "val"

    tb_log_dir = args.log_dir + "/" + args.title
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    print('Start Loading!')
    glove_file_path = './data/AI2thor_Combine_Dataset/det_feature_eval.hdf5'
    glove_file = hdf5_to_dict(glove_file_path)
    print('Loading Success!')

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.save_model_dir)]
    checkpoints = [
        (f, int(s[-3]))
        for (f, s) in checkpoints
        if len(s) >= 4 and f.startswith(args.title)
    ]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0
    for (f, train_ep) in tqdm(checkpoints, desc="Checkpoints."):

        model = os.path.join(args.save_model_dir, f)
        args.load_model = model

        # run eval on model
        args.test_or_val = "val"
        main_eval(args, create_shared_model, init_agent, glove_file)

        # check if best on val.
        with open(args.results_json, "r") as f:
            results = json.load(f)

        if results["success"] > best_performance_on_val:
            best_model_on_val = model
            best_performance_on_val = results["success"]

        log_writer.add_scalar("val/success", results["success"], train_ep)
        log_writer.add_scalar("val/spl", results["spl"], train_ep)

        # run on test.
        args.test_or_val = "test"
        main_eval(args, create_shared_model, init_agent, glove_file)
        with open(args.results_json, "r") as f:
            results = json.load(f)

        log_writer.add_scalar("test/success", results["success"], train_ep)
        log_writer.add_scalar("test/spl", results["spl"], train_ep)

    args.record_route = True
    args.test_or_val = "test"
    args.load_model = best_model_on_val
    main_eval(args, create_shared_model, init_agent, glove_file)

    with open(args.results_json, "r") as f:
        results = json.load(f)

    print(
        tabulate(
            [
                ["SPL >= 1:", results["GreaterThan/1/spl"]],
                ["Success >= 1:", results["GreaterThan/1/success"]],
                ["SPL >= 5:", results["GreaterThan/5/spl"]],
                ["Success >= 5:", results["GreaterThan/5/success"]],
            ],
            headers=["Metric", "Result"],
            tablefmt="orgtbl",
        )
    )

    print("Best model:", args.load_model)


if __name__ == "__main__":
    main()