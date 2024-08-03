import os
import random
import pickle
import argparse
import json
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(dataset_path):
    pkl_dump_dir = dataset_path
    df1 = open(pkl_dump_dir + "df.pkl", "rb")
    inf = pickle.load(df1, encoding='iso-8859-1')  # 读取pkl文件的内容
    print(inf)

    # 20news-coarse
    #labels = ['soc', 'talk', 'rec', 'sci', 'comp','misc', 'alt']
    #nyt-coarse
    labels = ['arts', 'business', 'politics', 'science', 'sports']
    #nyt-fine
    # labels = ['abortion', 'baseball', 'basketball','cosmos','dance','economy','energy_companies',
    #           'environment','federal_budget','football','gay_rights','golf',
    #           'gun_control','hockey','immigration','international_business','law_enforcement',
    #           'military','movies','music','soccer','stocks_and_bonds','surveillance',
    #           'television','tennis','the_affordable_care_act']
    df2 = open( pkl_dump_dir+"trans1/df.pkl", "wb")
    for i in range(0,len(df2)):
        if (i % 10 == 0):
            ori_label = inf['label'][i]
            a = random.choice(labels)
            inf['label'][i] = a
    pickle.dump(inf, df2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/nyt/coarse/')
    parser.add_argument('--gpu_id', type=str, default="1")
    args = parser.parse_args()
    if args.gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(dataset_path=args.dataset_path)