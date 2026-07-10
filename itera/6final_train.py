# -*- coding: utf8 -*-
# 这一段是实体概念扩展
import logging
import sys
import subprocess
import time
import os


if __name__ == '__main__':
    start_time = time.time()

    itera = 8   # g
    with open('output_LHd_targeted.txt', 'a+', encoding='utf8',errors='ignore') as t:
        t.write('itera: ' + str(itera) + '\n')

    file1 = open("train_all.csv", "r", encoding='gbk',errors='ignore')
    file2 = open("./datasets/TextClassification/LHd_targeted/train.csv", "w", encoding='gbk')

    s = file1.read()
    w = file2.write(s)

    file1.close()
    file2.close()

    for i in {100, 110, 120, 130, 140}:

        cmd = "python fewshot1.py --result_file ./output_LHd_targeted.txt --dataset LHd_targeted  --template_id 0  --seed  " + str(i) + " --shot 20 --verbalizer tkpt"
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.communicate()
        subprocess.run(cmd, shell=True, check=True)

        print("train seed {}:preprocess down".format(i))

        time.sleep(3)

    end_time = time.time()
    run_time = (end_time - start_time) / 5 * 3

    # for i in range(1, itera + 1):
    #     os.remove("LHd_targeted_label" + str(i) + ".csv")
    os.remove("result.csv")
    os.remove("result1.csv")
    os.remove("train_all.csv")
    # os.remove("LHd_targeted_prelabel_0.csv")
    # os.remove("ckpts/LHd_targeted_tem0.ckpt")

    with open('time.csv', 'a+', encoding='utf8', errors='ignore') as t:
        t.write('LHd_targeted, step6 runtime: ' + str(run_time) + '\n\n')

