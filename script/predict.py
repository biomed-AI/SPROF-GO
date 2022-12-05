import os, datetime, argparse, re
from tqdm import tqdm
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import DataLoader
import gc
from utils import *
from model import *
from label_diffusion import *

############ Set to your own path! ############
ProtTrans_path = "/home/yuanqm/tools/Prot-T5-XL-U50"
###############################################

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/model/"

Max_repr = np.load(script_path + "utility_files/ProtTrans_repr_max.npy")
Min_repr = np.load(script_path + "utility_files/ProtTrans_repr_min.npy")


def feature_extraction(ID_list, seq_list, outpath, feat_bs, device):
    feat_path = outpath + "ProtTrans_repr/"
    os.makedirs(feat_path, exist_ok = True)

    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
    gc.collect()

    # Load the model into CPU/GPU and switch to inference mode
    model = model.to(device)
    model = model.eval()

    # Extract feature of one batch each time
    for i in tqdm(range(0, len(ID_list), feat_bs)):
        if i + feat_bs <= len(ID_list):
            batch_ID_list = ID_list[i:i + feat_bs]
            batch_seq_list = seq_list[i:i + feat_bs]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]

        # Load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", " ".join(list(sequence))) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into GPU if avilabile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens = True, padding = True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extract sequence features and load it into CPU
        with torch.no_grad():
            embedding = model(input_ids = input_ids, attention_mask = attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            seq_emd = (seq_emd - Min_repr) / (Max_repr - Min_repr) # Min-Max normalization
            np.save(feat_path + batch_ID_list[seq_num], seq_emd)


def predict(ID_list, test_fasta, config, outpath, pred_bs, device):
    test_dataset = FunDataset(ID_list, outpath)
    test_dataloader = DataLoader(test_dataset, batch_size = pred_bs, collate_fn = test_dataset.collate_fn, shuffle = False, drop_last = False, num_workers = 4)

    predictions = []
    for task in ["MF", "BP", "CC"]:
        print("predicting {}...".format(task))

        # Load SPROF-GO models
        models = []
        for i in range(5):
            model = SPROF_GO(task, config["feature_dim"], config["hidden_dim"], config["num_emb_layers"], config["num_heads"], config["dropout"], device).to(device)
            state_dict = torch.load(model_path + '{}/{}_model_{}.ckpt'.format(task, task, i), device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        # Make predictions
        test_pred = []
        for batch in tqdm(test_dataloader):
            protein_feat, protein_mask = [x.to(device) for x in batch]
            with torch.no_grad():
                preds = [model(protein_feat, protein_mask) for model in models]
                preds = torch.stack(preds, 0).mean(0) # average predictions from 5 models
                test_pred.append(preds.detach().cpu().numpy())
        test_pred = np.concatenate(test_pred) # initial prediction

        # Label diffusion
        diffusion_pred = LabelDiffusion(task, test_pred, test_fasta, outpath) # final prediction
        predictions.append(np.round(diffusion_pred, decimals = 3))

    return predictions



def main(run_id, seq_info, outpath, topK, feat_bs, pred_bs, save_feat, gpu):
    fasta, ID_list, seq_list = seq_info
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    print("\n######## Feature extraction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    feature_extraction(ID_list, seq_list, outpath, feat_bs, device)

    print("\n######## Feature extraction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    print("\n######## Prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    predictions = predict(ID_list, fasta, NN_config, outpath, pred_bs, device)

    print("\n######## Prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    with open(script_path + "utility_files/idx_to_GO.pkl", "rb") as f:
        idx_to_GO = pickle.load(f)
    export_predictions(run_id, outpath, ID_list, predictions, idx_to_GO, topK)

    # Clean up
    if not save_feat:
        os.system("rm -rf {}ProtTrans_repr".format(outpath))
    os.system("rm {}test.fa".format(outpath))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type = str, help = "Input fasta file")
    parser.add_argument("--outpath", type = str, help = "Output path to save intermediate features and final predictions")
    parser.add_argument("--top", type = int, default=20, help = "Besides the full predictions, also show the terms with top K predictive scores")
    parser.add_argument("--feat_bs", type = int, default=8, help = "Batch size for ProtTrans feature extraction")
    parser.add_argument("--pred_bs", type = int, default=8, help = "Batch size for SPROF-GO prediction")
    parser.add_argument("--save_feat", action = "store_true", help = "Save intermediate ProtTrans features")
    parser.add_argument("--gpu", action = "store_true", help = "Use GPU for feature extraction and SPROF-GO prediction")

    args = parser.parse_args()
    outpath = args.outpath.rstrip("/") + "/"

    run_id = args.fasta.split("/")[-1].split(".")[0]
    seq_info = process_fasta(args.fasta, args.outpath)

    if seq_info == -1:
        print("The format of your input fasta file is incorrect! Please check!")
    elif seq_info == 1:
        print("Too much sequences! Up to {} sequences are supported each time!".format(MAX_INPUT_SEQ))
    else:
        main(run_id, seq_info, outpath, args.top, args.feat_bs, args.pred_bs, args.save_feat, args.gpu)
