# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

start_time = time.time()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    l = ['LHd']#
    batch_sizes = {16}
    learning_rates = {'4e-5'}
    shots = {5}
    seeds = {i for i in range(0,100)}
    template_id = {0}
    verbalizer = {'tkpt'}
    for n, t, j, i, k, m, v in product(l,template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):
        time_start = time.time()
        cmd = (
            f"python fewshot.py --result_file ./al_result.txt "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v} "
            f"--label_result ./out_label_text/{n}/train_seed{j}.csv "
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")
        time_end = time.time()
        print("run_time: "+str(time_end - time_start)+"秒")
        time.sleep(2)


end_time = time.time()
print("run_time: "+str(end_time - start_time)+"秒")
time.sleep(2)