import scipy.sparse as ssp
from scipy.sparse.linalg import inv
import numpy as np
import os, pickle, time

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
dataset_path = script_path + "utility_files/label_diffusion/"


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def jaccard(Q):
    Co = Q.dot(Q.T)
    CLEN = Q.sum(axis=1) # numpy matrix, shape = (N, 1)

    # original implementation: J = Co / (CLEN + CLEN.T - Co + 1)
    nonzero_mask = Co.astype('bool')
    denominator = nonzero_mask.multiply(CLEN) + nonzero_mask.multiply(CLEN.T) - Co + nonzero_mask

    J = sparse_divide_nonzero(Co, denominator)
    return J


def compute_L(Q):
    JQ = jaccard(Q).multiply(Q)
    degree = 1.0 / JQ.sum(axis=1) # shape = (N, 1)
    Q_star = 0.5 * (JQ.multiply(degree) + JQ.multiply(degree.T))

    degree = Q_star.sum(axis=1)
    D = ssp.spdiags(degree.T, 0, Q.shape[0], Q.shape[0])
    L = D - Q_star

    return L


def homology_matrix(fasta, cutoff = 0.1):
    ID = []
    with open(fasta, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID.append(line[1:-1])

    ID2idx = dict(zip(ID, range(len(ID))))

    os.system("{}diamond makedb --in {} -d {} --quiet".format(script_path, fasta, fasta))
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {} --very-sensitive --quiet -p 8'.format(script_path, fasta, fasta, fasta + ".tsv"))

    with open(fasta + ".tsv", "r") as f:
        out = f.readlines()

    os.system("rm {}.*".format(fasta))

    homology = {}
    for line in out:
        fields = line.strip().split()
        query_idx = ID2idx[fields[0]]
        subject_idx = ID2idx[fields[1]]
        identity = float(fields[2]) / 100

        if query_idx not in homology:
            homology[query_idx] = {}
        if subject_idx in homology[query_idx]:
            homology[query_idx][subject_idx] = max(homology[query_idx][subject_idx], identity)
        else:
            homology[query_idx][subject_idx] = identity

        # handle symmetry
        if subject_idx not in homology:
            homology[subject_idx] = {}
        if query_idx in homology[subject_idx]:
            homology[subject_idx][query_idx] = max(homology[subject_idx][query_idx], identity)
        else:
            homology[subject_idx][query_idx] = identity

    row = []
    col = []
    data = []
    for i in homology:
        for j in homology[i]:
            val = homology[i][j]

            if i == j:
                val = 1.0
            if val < cutoff:
                continue

            row.append(i)
            col.append(j)
            data.append(val)

    data = np.array(data)
    graph = ssp.csr_matrix((data, (row, col)), shape = (len(ID), len(ID)))

    return graph


def LabelDiffusion(task, initial_pred, test_fasta, outpath, lamda = 1, identity_cutoff = 0.1):
    '''
    This is an efficient implementation of the label diffusion algorithm
    by S2F, using DIAMOND and the sparse matrix throughout the caculation.
    '''
    print("Start diffusion...")
    start = time.time()

    # prepare the training set
    with open(dataset_path + task + "_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(dataset_path + task + "_valid.pkl", "rb") as f:
        valid_dataset = pickle.load(f)
    train_dataset = {**train_dataset, **valid_dataset}

    train_fasta = dataset_path + task + "_train+valid.fa"
    if not os.path.exists(train_fasta):
        fasta_file = ""
        for ID in train_dataset:
            fasta_file += (">" + ID + "\n" + train_dataset[ID][0] + "\n")
        with open(train_fasta, "w") as f:
            f.write(fasta_file)


    # search for hits/seeds in the training set for the test proteins
    os.system("{}diamond makedb --in {} -d {} --quiet".format(script_path, train_fasta, train_fasta))
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {}testVStrain.tsv --very-sensitive --quiet -p 8'.format(script_path, train_fasta, test_fasta, outpath))
    os.system("rm {}.dmnd".format(train_fasta))

    with open(outpath + "testVStrain.tsv", "r") as f:
        testVStrain = f.readlines()

    train_seed_ID = set()
    for line in testVStrain:
        fields = line.strip().split()
        subject_id = fields[1]
        identity = float(fields[2]) / 100
        if identity > identity_cutoff:
            train_seed_ID.add(subject_id)

    os.system("rm {}testVStrain.tsv".format(outpath))

    if len(initial_pred) == 1 and len(train_seed_ID) == 0: # only one test seq and no train seed found
        end = time.time()
        print("Diffusion done! Cost {}s.".format(end - start))
        return initial_pred

    train_seed_ID_list = []
    train_seed_fasta = ""
    for ID in train_dataset: # keep the original ID order
        if ID in train_seed_ID:
            train_seed_ID_list.append(ID)
            train_seed_fasta += (">" + ID + "\n" + train_dataset[ID][0] + "\n")

    with open(outpath + "train_seed_seq.fa", "w") as f:
        f.write(train_seed_fasta)


    # get the homology matrix Q for the test proteins + seeds in the training set
    os.system("cat {}train_seed_seq.fa {} > {}train_seed_and_test.fa".format(outpath, test_fasta, outpath))
    os.system("rm {}train_seed_seq.fa".format(outpath))

    Q = homology_matrix(outpath + "train_seed_and_test.fa")
    os.system("rm {}train_seed_and_test.fa".format(outpath))


    # get labels of train seeds
    row = []
    col = []
    for i, ID in enumerate(train_seed_ID_list):
        label_idx = train_dataset[ID][1]
        row += [i] * len(label_idx)
        col += label_idx
    data = [1] * len(row)

    label_size = {"MF":790, "BP":4766, "CC":667}[task]
    train_seed_label = ssp.csr_matrix((data, (row, col)), shape = (len(train_seed_ID_list), label_size))


    # Label Diffusion
    initial_pred = ssp.csr_matrix(initial_pred)
    initial_pred = ssp.vstack([train_seed_label, initial_pred])

    L = compute_L(Q)
    IlambdaL = ssp.identity(Q.shape[0]) + L.multiply(lamda)
    kernel = inv(IlambdaL.tocsc())[len(train_seed_ID_list):]
    diffusion_pred = kernel.dot(initial_pred).toarray()

    end = time.time()
    print("Diffusion done! Cost {}s.".format(end - start))

    return diffusion_pred
