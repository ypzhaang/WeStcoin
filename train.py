import pickle
import argparse
import json
import gc
import math
from util import *
from caculate import *
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import ModelCheckpoint
from collections import defaultdict
from gensim.models import word2vec
from keras_han.HAN_NL_Im3 import HAN
from nltk.corpus import stopwords
import os
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # 导入计算混淆矩阵的包


def main(dataset_path, print_flag=True):
    def train_word2vec(df, dataset_path):
        def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                           mode='skipgram',
                           min_word_count=2,
                           context=5):
            num_workers = 15  # Number of threads to run in parallel
            downsampling = 1e-3  # Downsample setting for frequent words
            print('Training Word2Vec model...')
            sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
            if mode == 'skipgram':
                sg = 1
                print('Model: skip-gram')
            elif mode == 'cbow':
                sg = 0
                print('Model: CBOW')
            embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                                sg=sg,
                                                size=size_features,
                                                min_count=min_word_count,
                                                window=context,
                                                sample=downsampling)
            embedding_model.init_sims(replace=True)
            embedding_weights = np.zeros((len(vocabulary_inv) + 1, size_features))
            embedding_weights[0] = 0
            for i, word in vocabulary_inv.items():
                if word in embedding_model:
                    embedding_weights[i] = embedding_model[word]
                else:
                    embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)

            return embedding_weights

        tokenizer = fit_get_tokenizer(df.sentence, max_words=150000)
        print("Total number of words: ", len(tokenizer.word_index))
        tagged_data = tokenizer.texts_to_sequences(df.sentence)
        vocabulary_inv = {}
        for word in tokenizer.word_index:
            vocabulary_inv[tokenizer.word_index[word]] = word
        embedding_mat = get_embeddings(tagged_data, vocabulary_inv)
        pickle.dump(tokenizer, open(dataset_path + "tokenizer.pkl", "wb"))
        pickle.dump(embedding_mat, open(dataset_path + "embedding_matrix.pkl", "wb"))

    def preprocess(df, word_cluster):
        print("Preprocessing data..")
        stop_words = set(stopwords.words('english'))
        stop_words.add('would')
        word_vec = {}
        for index, row in df.iterrows():
            if index % 10000 == 0:
                print("Finished rows: " + str(index) + " out of " + str(len(df)))
            line = row["sentence"]
            words = line.strip().split()
            new_words = []
            for word in words:
                try:
                    vec = word_vec[word]
                except:
                    vec = get_vec(word, word_cluster, stop_words)
                    if len(vec) == 0:
                        continue
                    word_vec[word] = vec
                new_words.append(word)
            df["sentence"][index] = " ".join(new_words)
        return df, word_vec

    def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
        def argmax_label(i, count_dict):
            counts = {}  # 计算每一个类别的总数
            for seed in labels:
                counts[seed] = 0
            maxi = 0
            max_label = None
            for l in count_dict:##l是类别
                counts[l] = 0
                count = 0
                for t in count_dict[l]:#t是属于l类的每一个关键词
                    count += count_dict[l][t]
                counts[l] = count###########count 计算所属每一类seed出现次数之和
                if count > maxi:#每次对count进行判断，给max_label进行复制
                    maxi = count
                    max_label = l
            summary = sum(counts.values())  # 定义到函数外，只需计算一次
            for k, v in counts.items():
                counts[k] = v / summary
            prob = []
            for k, v in counts.items():
                prob.append(v)
            return max_label, prob

        i = 1
        yp = []
        y = []
        X = []
        pseudo_labels = []
        y_true = []
        index_word = {}
        for w in tokenizer.word_index:
            index_word[tokenizer.word_index[w]] = w
        for index, row in df.iterrows():
            line = row["sentence"]
            label = row["label"]
            tokens = tokenizer.texts_to_sequences([line])[0]
            words = []
            for tok in tokens:
                words.append(index_word[tok])
            count_dict = {}
            flag = 0
            for l in labels:
                seed_words = set()
                for w in label_term_dict[l]:
                    seed_words.add(w)
                int_labels = list(set(words).intersection(seed_words))
                if len(int_labels) == 0:
                    continue
                for word in words:
                    if word in int_labels:
                        flag = 1
                        try:
                            temp = count_dict[l]
                        except:
                            count_dict[l] = {}
                        try:
                            count_dict[l][word] += 1
                        except:
                            count_dict[l][word] = 1
            if flag:
                lbl , lbp = argmax_label(i, count_dict)
                i = i + 1
                if not lbp:
                    continue
                pseudo_labels.append(lbl)
                y.append(lbp)
                X.append(line)
                y_true.append(label)

        return X, pseudo_labels, y, y_true

    def train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, dataset_path):
        print("Going to train classifier..")
        basepath = dataset_path
        model_name = "conwea"
        dump_dir = basepath + "models/" + model_name + "/"
        tmp_dir = basepath + "checkpoints/" + model_name + "/"
        os.makedirs(dump_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        max_sentence_length = 100
        max_sentences = 15
        max_words = 20000
        tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))


        X, pseudo_labels, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)
        y_one_hot = make_one_hot(y_true, label_to_index)
        print("Fitting tokenizer...")
        print("Splitting into train, dev...")
        X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                          max_sentences=max_sentences,
                                                          max_sentence_length=max_sentence_length,
                                                          max_words=max_words)
        X_train, Ts_train, X_val, Ts_val = create_train_dev(X, labels=y, tokenizer=tokenizer,
                                                            max_sentences=max_sentences,
                                                            max_sentence_length=max_sentence_length,
                                                            max_words=max_words)
        pseudo_onehot = make_one_hot(pseudo_labels, label_to_index)
        X_train, pseudo_train, X_val, pseudo_val = create_train_dev(X, labels=pseudo_onehot, tokenizer=tokenizer,
                                                            max_sentences=max_sentences,
                                                            max_sentence_length=max_sentence_length,
                                                            max_words=max_words)
        PTs_train_array = np.array(Ts_train)
        PTs_val_array = np.array(Ts_val)
        
        H = cacul_D1(y_train[0], train_labels)
        S = cacul_D2(len(y[0]), y_train[0], train_labels)
        train_labels = get_from_one_hot(pseudo_train, index_to_label)#得到y_train的labels列表
        val_labels = get_from_one_hot(pseudo_val, index_to_label)
        train_S = np.array(index(train_labels, S))
        val_S = np.array(index(val_labels, S))
        train_H = np.array(index(train_labels, C))
        val_H = np.array(index(val_labels, C))
        train_mul_C = np.multiply(train_S, train_H)
        val_mul_C = np.multiply(val_S, val_H)
        print("Creating Embedding matrix...")
        embedding_matrix = pickle.load(open(dataset_path + "embedding_matrix.pkl", "rb"))
        print("Initializing model...")
        model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                    embedding_matrix=embedding_matrix)
        print("Compiling model...")
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print("model fitting - Hierachical attention network...")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                             verbose=1, save_weights_only=True, save_best_only=True)
        test_train = np.ones((len(X_train), 200))
        test_val = np.ones((len(X_val),200))
        model.fit([X_train, PTs_train_array, train_mul_C], [y_train, y_train], validation_data=([X_val, PTs_val_array, val_mul_C], [y_val, y_val]), nb_epoch=10, batch_size=256, callbacks=[es, mc])
        print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
        X_all = prep_data(texts=df["sentence"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                          tokenizer=tokenizer)
        df = pickle.load(open(pkl_dump_dir + "df.pkl", "rb"))
        y_true_all = df["label"]
        test_mul_C = np.ones((len(X_all), len(y_train[0])))
        pred, out = model.predict([X_all, test_mul_C, test_mul_C])
        pred_labels = get_from_one_hot(pred, index_to_label)
        out_labels = get_from_one_hot(out, index_to_label)
        C = confusion_matrix(y_true_all, pred_labels)
        print(classification_report(y_true_all, pred_labels))
        print("Dumping the model...")
        model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
        model.save(dump_dir + "model_" + model_name + ".h5")
        return out_labels

    def expand_seeds(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index, index_to_word,
                     inv_docfreq, docfreq, it, n1, doc_freq_thresh=5):
        def get_rank_matrix(docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index, term_count,
                            word_to_index, doc_freq_thresh):
            E_LT = np.zeros((label_count, term_count))
            components = {}
            for l in label_docs_dict:
                components[l] = {}
                docs = label_docs_dict[l]
                docfreq_local = calculate_doc_freq(docs)
                vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
                X = vect.transform(docs)
                X_arr = X.toarray()
                rel_freq = np.sum(X_arr, axis=0) / len(docs)
                names = vect.get_feature_names()
                for i, name in enumerate(names):
                    try:
                        if docfreq_local[name] < doc_freq_thresh:
                            continue
                    except:
                        continue
                    E_LT[label_to_index[l]][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[
                        name] * np.tanh(rel_freq[i])
                    components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                           "idf": inv_docfreq[name],
                                           "rel_freq": np.tanh(rel_freq[i]),
                                           "rank": E_LT[label_to_index[l]][word_to_index[name]]}
            return E_LT, components

        def disambiguate(label_term_dict, components):
            new_dic = {}
            for l in label_term_dict:
                all_interp_seeds = label_term_dict[l]
                seed_to_all_interp = {}
                disambiguated_seed_list = []
                for word in all_interp_seeds:
                    temp = word.split("$")
                    if len(temp) == 1:
                        disambiguated_seed_list.append(word)
                    else:
                        try:
                            seed_to_all_interp[temp[0]].add(word)
                        except:
                            seed_to_all_interp[temp[0]] = {word}

                for seed in seed_to_all_interp:
                    interpretations = seed_to_all_interp[seed]
                    max_interp = ""
                    maxi = -1
                    for interp in interpretations:
                        try:
                            if components[l][interp]["rank"] > maxi:
                                max_interp = interp
                                maxi = components[l][interp]["rank"]
                        except:
                            continue
                    disambiguated_seed_list.append(max_interp)
                new_dic[l] = disambiguated_seed_list
            return new_dic

        def expand(E_LT, index_to_label, index_to_word, it, label_count, n1, old_label_term_dict, label_docs_dict):
            word_map = {}
            zero_docs_labels = set()
            for l in range(label_count):
                if not np.any(E_LT):
                    continue
                elif len(label_docs_dict[index_to_label[l]]) == 0:
                    zero_docs_labels.add(index_to_label[l])
                else:
                    n = min(n1 * (it), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
                    inds_popular = E_LT[l].argsort()[::-1][:n]
                    for word_ind in inds_popular:
                        word = index_to_word[word_ind]
                        try:
                            temp = word_map[word]
                            if E_LT[l][word_ind] > temp[1]:
                                word_map[word] = (index_to_label[l], E_LT[l][word_ind])
                        except:
                            word_map[word] = (index_to_label[l], E_LT[l][word_ind])

            new_label_term_dict = defaultdict(set)
            for word in word_map:
                label, val = word_map[word]
                new_label_term_dict[label].add(word)
            for l in zero_docs_labels:
                new_label_term_dict[l] = old_label_term_dict[l]
            return new_label_term_dict

        label_count = len(label_to_index)
        term_count = len(word_to_index)
        label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

        E_LT, components = get_rank_matrix(docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                                           term_count, word_to_index, doc_freq_thresh)

        if it == 0:
            print("Disambiguating seeds..")
            label_term_dict = disambiguate(label_term_dict, components)
        else:
            print("Expanding seeds..")
            label_term_dict = expand(E_LT, index_to_label, index_to_word, it, label_count, n1, label_term_dict,
                                     label_docs_dict)
        return label_term_dict, components

    pkl_dump_dir = dataset_path
    df = pickle.load(open(pkl_dump_dir + "df_contextualized_trans.pkl", "rb"))
    word_cluster = pickle.load(open(pkl_dump_dir +"word_cluster_map.pkl", "rb"))
    with open(pkl_dump_dir + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    label_term_dict = add_all_interpretations(label_term_dict, word_cluster)
    print_label_term_dict(label_term_dict, None, print_components=False)
    labels = list(set(label_term_dict.keys()))
    label_to_index = {'politics': 0, 'arts': 1, 'sports': 2, 'science': 3, 'business': 4}
    index_to_label = {0: 'politics', 1: 'arts', 2: 'sports', 3: 'science', 4: 'business'}
    # label_to_index = {'football': 0, 'law_enforcement': 1, 'cosmos': 2, 'federal_budget': 3, 'immigration': 4,
    #                   'energy_companies': 5, 'international_business': 6, 'stocks_and_bonds': 7, 'basketball': 8,
    #                   'soccer': 9, 'surveillance': 10, 'economy': 11, 'tennis': 12, 'military': 13, 'golf': 14,
    #                   'music': 15,
    #                   'abortion': 16, 'gay_rights': 17, 'television': 18, 'dance': 19, 'movies': 20,
    #                   'gun_control': 21, 'baseball': 22, 'environment': 23, 'the_affordable_care_act': 24, 'hockey': 25}
    # index_to_label = {0: 'football', 1: 'law_enforcement', 2: 'cosmos', 3: 'federal_budget', 4: 'immigration',
    #                   5: 'energy_companies', 6: 'international_business', 7: 'stocks_and_bonds', 8: 'basketball',
    #                   9: 'soccer', 10: 'surveillance', 11: 'economy', 12: 'tennis', 13: 'military', 14: 'golf',
    #                   15: 'music',
    #                   16: 'abortion', 17: 'gay_rights', 18: 'television', 19: 'dance', 20: 'movies',
    #                   21: 'gun_control', 22: 'baseball', 23: 'environment', 24: 'the_affordable_care_act', 25: 'hockey'}
    # label_to_index = {'soc': 0, 'talk': 1, 'rec': 2, 'sci': 3, 'comp': 4, 'misc': 5, 'alt': 6}
    # index_to_label = {0: 'soc', 1: 'talk', 2: 'rec', 3: 'sci', 4: 'comp', 5: 'misc', 6: 'alt'}
    df, word_vec = preprocess(df, word_cluster)
    del word_cluster
    gc.collect()
    word_to_index, index_to_word = create_word_index_maps(word_vec)
    docfreq = calculate_df_doc_freq(df)
    inv_docfreq = calculate_inv_doc_freq(df, docfreq)

    train_word2vec(df, dataset_path)
    for i in range(10):#6 times iteration
        print("ITERATION: ", i)
        pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, dataset_path)
        label_term_dict, components = expand_seeds(df, label_term_dict, pred_labels, label_to_index, index_to_label,
                                                   word_to_index, index_to_word, inv_docfreq, docfreq, i, n1=5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/nyt/coarse/')
    parser.add_argument('--gpu_id', type=str, default="1")
    args = parser.parse_args()
    if args.gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(dataset_path=args.dataset_path)
