import os
import os.path as osp
from os.path import join as pjoin
import argparse
import random
from tqdm import tqdm
import numpy as np

import codecs as cs


from multiprocessing import Pool, Queue, Manager


def worker(name, root_dir, min_motion_length, data_queue):
    data_dict = {}
    try:
        motion = np.load(pjoin(root_dir, "new_joint_vecs", name + ".npy"), allow_pickle=True)
        if (len(motion)) < min_motion_length or (len(motion) >= 200):
            print(f"skip {name}, len {len(motion)}")
            return

        # raw_motion = np.load(pjoin(root_dir, "new_joints", name + ".npy"), allow_pickle=True) 
        # if len(raw_motion) != len(motion):
        #     print(name, len(motion), len(raw_motion))
        #     return

        text_data = []
        flag = False
        with (cs.open(pjoin(root_dir, "texts", name + ".txt")) as f):
            for j, line in enumerate(f.readlines()):
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = tokens
                if f_tag == 0.0 and to_tag == 0.0:
                    flag = True
                    text_data.append(text_dict)
                else:
                    try:
                        n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                        # n_raw_motion = raw_motion[int(f_tag * 20):int(to_tag * 20)]
                        if (len(n_motion)) < min_motion_length or ((len(n_motion) >= 200)):
                            continue
                        new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)
                        while new_name in data_dict:
                            new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)

                        data_dict[new_name] = {
                            "motion": n_motion,
                            "length": len(n_motion),
                            "text": [text_dict],
                        }
                    except:
                        print(f"error1! {name}")
                        pass

        if flag:
            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "text": text_data,
            }
    except:
        print(f"error2! {name}")
        pass

    data_queue.put(data_dict)

def process(data_dir, split, min_motion_length, args):
    print(f"processing {split}...")
    with cs.open(osp.join(data_dir, split+".txt"), "r") as f:
        id_list = [line.strip() for line in f.readlines()]
    results = {}

    with Manager() as manager:
        output_queue = manager.Queue()
        pool = Pool(processes=args.num_workers)

        with tqdm(total=len(id_list)) as pbar:
            for x in id_list:
                pool.apply_async(worker,
                                 (x, data_dir, min_motion_length, output_queue),
                                 callback=lambda _: pbar.update(1))

            pool.close()
            pool.join()

        while not output_queue.empty():
            ret = output_queue.get()
            results.update(ret)
    np.save(osp.join(data_dir, "data_"+split+".npy"), results)

def main(args):
    if args.dataset == "hml3d":
        data_dir = "./data/HumanML3D/"
        min_motion_length = 40
    elif args.dataset == "kit":
        data_dir = "./data/KIT-ML/"
        min_motion_length = 24
    else:
        raise ValueError(f"{args.dataset} not supported!")

    process(data_dir, "train", min_motion_length, args)
    process(data_dir, "val", min_motion_length, args)
    process(data_dir, "test", min_motion_length, args)

    print(f"dataset {args.dataset} preparation done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hml3d", required=True)
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()
    main(args)