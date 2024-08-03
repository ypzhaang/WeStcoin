import numpy as np


def cacul_D1(len(y[0]), y_train[0], train_labels):
    C = np.ones((len(y_train[0]), len(y_train[0])))  # c*c
    #complete here :distribution of class
    # A = np.array(
    #     [1526, 413, 41, 113, 57, 137, 456, 110, 998, 1256, 50, 270, 796, 59, 1013, 412, 29, 84, 194, 108, 326, 58,
    #      2214, 48, 114, 645])
    h = (A / len(y[0]))
    for i in range(len(y_train[0])):
        for j in range(len(y_train[0])):
            if i == j:
                C[i][j] = h[i]
            else:
                C[i][j] = 1
    np.savetxt("nyt-fine-H.txtx", C)
    return C


def cacul_D2(len(y[0]), y_train[0], train_labels):
    doc_summary = np.loadtxt('nyt_fine_doc_summary.txt')
    index_number = []  # class label for each doc
    for label in train_labels:
        #nyt-fine
        if label == "football":
            index_number.append(0)
        if label == "law_enforcement":
            index_number.append(1)
        if label == "cosmos":
            index_number.append(2)
        if label == "federal_budget":
            index_number.append(3)
        if label == "immigration":
            index_number.append(4)
        if label == "energy_companies":
            index_number.append(5)
        if label == "international_business":
            index_number.append(6)
        if label == "stocks_and_bonds":
            index_number.append(7)
        if label == "basketball":
            index_number.append(8)
        if label == "soccer":
            index_number.append(9)
        if label == "surveillance":
            index_number.append(10)
        if label == "economy":
            index_number.append(11)
        if label == "tennis":
            index_number.append(12)
        if label == "military":
            index_number.append(13)
        if label == "golf":
            index_number.append(14)
        if label == "music":
            index_number.append(15)
        if label == "abortion":
            index_number.append(16)
        if label == "gay_rights":
            index_number.append(17)
        if label == "television":
            index_number.append(18)
        if label == "dance":
            index_number.append(19)
        if label == "movies":
            index_number.append(20)
        if label == "gun_control":
            index_number.append(21)
        if label == "baseball":
            index_number.append(22)
        if label == "environment":
            index_number.append(23)
        if label == "the_affordable_care_act":
            index_number.append(24)
        if label == "hockey":
            index_number.append(25)
        #nyt-coarse:
        # if label == "sports":
        #     index_number.append(C[2])
        # if label == "politics":
        #     index_number.append(C[0])
        # if label == "business":
        #     index_number.append(C[4])
        # if label == "science":
        #     index_number.append(C[3])
        # if label == "arts":
        #     index_number.append(C[1])
        #20news-coarse:
        # if label == "soc":
        #     index_number.append(0)
        # if label == "talk":
        #     index_number.append(1)
        # if label == "rec":
        #     index_number.append(2)
        # if label == "sci":
        #     index_number.append(3)
        # if label == "comp":
        #     index_number.append(4)
        # if label == "misc":
        #     index_number.append(5)
        # if label == "alt":
        #     index_number.append(6)
    ###############################
    sum = np.zeros((len(y[0]), 200))  # sum for each class
    lens = np.zeros(len(y[0]))  # counts for each class
    ave = np.zeros((len(y[0]), 200))  # center point
    for i in index_number:
        sum[i] += doc_summary[i]
        lens[i] += 1
    for i in index_number:
        ave[i] = sum[i] / lens[i]
    # dist_inter
    dist_intra = np.zeros(len(y_train[0]))  
    for i, On in enumerate(doc_summary):  
        j = index_number[i]  
        dist = np.sqrt(np.sum(np.square(ave[j] - On)))  
        dist_intra[j] += dist
    # dist_intra
    dist_inter = np.zeros((len(y_train[0]), len(y_train[0])))  
    for i, a1 in enumerate(ave):
        for j, a2 in enumerate(ave):
            if i < j:
                dist_inter[i][j] = np.sqrt(np.sum(np.square(ave[i] - ave[j])))
    S = np.ones((len(y_train[0]), len(y_train[0])))  # 
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if i > j:
                S[i][j] = dist_intra[i] / (lens[i] * dist_inter[j][i])
            if i < j:
                S[i][j] = dist_intra[i] / (lens[i] * dist_inter[i][j])
    np.savetxt("nyt_fine_dist.txt", S)
    return S