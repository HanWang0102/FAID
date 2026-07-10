
import os
import time

start_time = time.time()

save_path1 = "LHd_label0.csv"
read_path1 = 'LHd_prelabel0.csv'
f_1 = open(save_path1, 'w', encoding='utf8',errors='ignore')
with open(read_path1, 'r', encoding='utf8',errors='ignore') as f:
    for ii, label in enumerate(f):
        label = int(label)
        label = label+1
        label = str(label)
        f_1.write('"' + label + '"'+'\n')
    f.close()
f_1.close()

read_path2 = './datasets/TextClassification/LHd_targeted/test.csv'
f_2 = open("LHd_test.csv", 'w', encoding='utf8', errors='ignore')
with open(read_path2, 'r', encoding='utf8',errors='ignore') as t:
    for ii, text in enumerate(t):
        text = text.split(',')
        text = ','.join(text[1:])
        f_2.write(text)
    t.close()
f_2.close()

with open('LHd_label0.csv', 'r', encoding='utf8', errors='ignore') as fa:
    with open('LHd_test.csv', 'r', encoding='utf8', errors='ignore') as fb:
        with open('LHd_pre0.csv', 'w', encoding='utf8', errors='ignore') as fc:
            for line in fa:
                fc.write(line.strip('\r\n'))
                fc.write(',')
                fc.write(fb.readline())
            fc.close()


save_path2 = "LHd_test0.csv"
save_path3 = "LHd_testall.csv"
read_path3 = 'LHd_pre0.csv'

f_w = open(save_path2, 'w', encoding='utf8',errors='ignore')
t_w = open(save_path3, 'w', encoding='utf8',errors='ignore')
with open(read_path3, 'r', encoding='utf8',errors='ignore') as f:
    for ii, text in enumerate(f):
        text = text.split(',')
        label = text[0]
        text = ','.join(text[1:]).replace('\n', '')
        if label == '"1"':
            with open(read_path2, 'r', encoding='utf8',errors='ignore') as t:
                for jj, text1 in enumerate(t):
                    if ii == jj:
                        text1 = text1.split(',')
                        label1 = text1[0]
                        text1 = ','.join(text1[1:]).replace('\n', '')
                        t_w.write(label1 + ',' + text1 + '\n')
            f_w.write(label + ',' + text + '\n')

f_w = open(save_path2, 'a', encoding='utf8',errors='ignore')
t_w = open(save_path3, 'a', encoding='utf8',errors='ignore')
with open(read_path3, 'r', encoding='utf8',errors='ignore') as t:
    for ii, text in enumerate(t):
        # print(text)
        text = text.split(',')
        label = text[0]
        text = ','.join(text[1:]).replace('\n', '')
        if label == '"2"':
            with open(read_path2, 'r', encoding='utf8',errors='ignore') as t:
                for jj, text1 in enumerate(t):
                    if ii == jj:
                        text1 = text1.split(',')
                        label1 = text1[0]
                        text1 = ','.join(text1[1:]).replace('\n', '')
                        t_w.write(label1 + ',' + text1 + '\n')
            f_w.write(label + ',' + text + '\n')

# os.remove("LHd_label0.csv")
# os.remove("LHd_test.csv")
# os.remove("LHd_pre0.csv")

# 划分数据集LHd（60train，60val）
read_path = 'LHd_test0.csv'
m = 0
n = 0

with open(read_path, 'r', encoding='utf8',errors='ignore') as f:
    for ii, text in enumerate(f):
        text = text.split(',')
        label = text[0]
        if label == '"1"':
            m += 1
        else:
            n += 1
print(m, n)
k = 0
j = m
for i in range(1, 11):
    save_path = "LHd_train" + str(i) + ".csv"
    f_w = open(save_path, 'w', encoding='utf8',errors='ignore')
    print('----------')
    with open(read_path, 'r', encoding='utf8',errors='ignore') as f:
        for ii, text in enumerate(f):
            text = text.split(',')
            label = text[0]
            text = ','.join(text[1:]).replace('\n', '')
            if k+60 <= m:
                if k <= ii < k+60:    #train1
                    f_w.write(label + ',' + text + '\n')
            elif m-k < 60:
                if k <= ii < m or 0 <= ii < 60-m+k:
                    f_w.write(label + ',' + text + '\n')

            if j+60 <= m+n:
                if j <= ii < j+60:   #train1
                    f_w.write(label + ',' + text + '\n')
            elif m+n-j < 60:
                if j <= ii < m+n or m <= ii < j + 60 - n:
                    f_w.write(label + ',' + text + '\n')
    if k+60 <= m:
        print("train{}:{}-{}".format(i, k, k + 60))
    elif m - k < 60:
        print("train{}:{}-{},{}-{}".format(i, k, m, 0, 60-m+k))

    if j + 60 <= m + n:
        print("train{}:{}-{}".format(i, j, j + 60))
    elif m + n - j < 60:
        print("train{}:{}-{},{}-{}".format(i, j, m + n, m, j + 60 - n))

    k += 60
    if k > m:
        k = k-m
    j += 60
    if j > m+n:
        j = j-n
end_time = time.time()
run_time = end_time - start_time
with open('time.csv', 'a+', encoding='utf8', errors='ignore') as t:
    t.write('LHd, step1-2 runtime: ' + str(run_time) + '\n')






