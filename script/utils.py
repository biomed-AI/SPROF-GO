import string, re
import numpy as np
import torch

MAX_INPUT_SEQ = 5000
label_size = {"MF":790, "BP":4766, "CC":667}

NN_config = {
    'feature_dim': 1024,
    'hidden_dim': 256,
    'num_emb_layers': 2,
    'num_heads': 8,
    'dropout': 0.1
}


class FunDataset:
    def __init__(self, ID_list, outpath):
        self.IDs = ID_list
        self.outpath = outpath
        self.feat_dim = NN_config['feature_dim']

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]
        protein_feat = torch.tensor(np.load(self.outpath + "ProtTrans_repr/" + ID + ".npy"), dtype = torch.float)
        return protein_feat

    def padding(self, batch):
        maxlen = max([protein_feat.shape[0] for protein_feat in batch])
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = torch.zeros(maxlen, self.feat_dim)
            padded_protein_feat[:protein_feat.shape[0]] = protein_feat
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = torch.zeros(maxlen)
            protein_mask[:protein_feat.shape[0]] = 1
            batch_protein_mask.append(protein_mask)

        return torch.stack(batch_protein_feat), torch.stack(batch_protein_mask)

    def collate_fn(self, batch):
        protein_feat, protein_mask = self.padding(batch) # [B, maxlen, hid], [B, maxlen]
        return protein_feat, protein_mask


def get_ID(name_item): # deal with IDs with different format
    name_item = name_item.split("|")
    ID = "_".join(name_item[0:min(2, len(name_item))])
    ID = re.sub(" ", "_", ID)
    return ID


def process_fasta(fasta_file, outpath):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID_list.append(get_ID(line[1:-1]))
        elif line[0] in string.ascii_letters:
            seq = line.strip().upper()
            seq_list.append(seq[0:min(5000, len(seq))]) # trim long sequence to 5000

    if len(ID_list) == len(seq_list):
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            new_fasta = "" # with processed IDs and seqs
            for i in range(len(ID_list)):
                new_fasta += (">" + ID_list[i] + "\n" + seq_list[i] + "\n")
            with open(outpath + "test.fa", "w") as f:
                f.write(new_fasta)

            return [outpath + "test.fa", ID_list, seq_list]
    else:
        return -1


def export_predictions(run_id, outpath, ID_list, predictions, idx_to_GO, topK):
    top_pred = ""
    all_pred = "GO term id and name:\n"
    for task in ["MF", "BP", "CC"]:
        all_pred += (task + ":\n")
        all_pred += ("; ".join([term[0] for term in idx_to_GO[task]]) + "\n")
        all_pred += ("; ".join([term[1] for term in idx_to_GO[task]]) + "\n")
    all_pred += "\n"

    for i in range(len(ID_list)):
        top_pred += (ID_list[i] + "\n")
        all_pred += (ID_list[i] + "\n")
        for j, task in enumerate(["MF", "BP", "CC"]):
            top_pred += (task + ":\n")
            topK_pred = sorted(list(zip(predictions[j][i], idx_to_GO[task])), reverse = True)[0:min(topK, label_size[task])]
            for pred in topK_pred:
               top_pred += (pred[1][0] + " | " + pred[1][1] + " | " + str(pred[0]) + "\n")

            all_pred += (task + ":\n")
            all_pred += ("; ".join([str(score) for score in predictions[j][i]]) + "\n")
        top_pred += "\n"
        all_pred += "\n"

    with open(outpath + run_id + "_top_preds.txt", "w") as f:
        f.write(top_pred)
    with open(outpath + run_id + "_all_preds.txt", "w") as f:
        f.write(all_pred)

    print("Results are saved in {}_top_preds.txt and {}_all_preds.txt under {}\n".format(run_id, run_id, outpath))
