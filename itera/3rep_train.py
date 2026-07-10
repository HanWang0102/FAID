# -*- coding: utf8 -*-

import logging
import sys
import subprocess
import time
import os

start_time = time.time()

if __name__ == '__main__':

    itera = 7# g
    with open('output_LHd.txt', 'a+', encoding='utf8', errors='ignore') as t:
        t.write('train: ' + str(itera) + '\n')

    file1 = open("LHd_testall.csv", "r", encoding='utf8', errors='ignore')
    file2 = open("datasets/TextClassification/LHd_targeted/test.csv", "w", encoding='utf8', errors='ignore')

    m = file1.read()
    n = file2.write(m)

    file1.close()
    file2.close()

    for i in range(1, itera + 1):
        file3 = open("LHd_train" + str(i) + ".csv", "r", encoding='utf8',errors='ignore')
        file4 = open("datasets/TextClassification/LHd_targeted/train.csv", "w", encoding='utf8', errors='ignore')

        s = file3.read()
        w = file4.write(s)

        file3.close()
        file4.close()

        cmd = "python fewshot1.py --result_file ./output_LHd.txt --dataset LHd  --template_id 0  --seed 25 --shot 30  --verbalizer tkpt"
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.communicate()
        subprocess.run(cmd, shell=True, check=True)

        file5 = open("LHd_prelabel_0.csv", "r", encoding='utf8', errors='ignore')
        file6 = open("LHd_prelabel" + str(i) + ".csv", "w", encoding='utf8',errors='ignore')

        a = file5.read()
        b = file6.write(a)

        file5.close()
        file6.close()
        print("train {}:preprocess down".format(i))

        time.sleep(3)

    end_time = time.time()
    run_time = end_time - start_time
    with open('time.csv', 'a+', encoding='utf8', errors='ignore') as t:
        t.write('LHd, step3 runtime: ' + str(run_time) + '\n')




