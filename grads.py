import torch
from datasets import load_from_disk
import geneformer
from geneformer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu
from geneformer import emb_extractor as ee
import numpy as np
from torch import nn
import pickle

# Taken from .../geneformer/perturber_utils.py
# =========================================================================================================

# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens, dim=1):
    # create a mask tensor based on padding lengths
    mask = torch.arange(embs.size(dim), device=embs.device) < original_lens.unsqueeze(1)
    # if embs.dim() == 3:
        # fill the masked positions in embs with zeros
    masked_embs = embs.masked_fill(~mask.unsqueeze(2), 0.0)
    masked_embs.retain_grad()

    # compute the mean across the non-padding dimensions
    mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

    # elif embs.dim() == 2:
    #     masked_embs = embs.masked_fill(~mask, 0.0)
    #     mean_embs = masked_embs.sum(dim) / original_lens.float()
    mean_embs.retain_grad()
    return mean_embs

# =========================================================================================================

def one_hot(lis):
    nd_arr = np.array(lis)
    tens = torch.zeros([len(lis), 2], dtype=torch.float64)
    act_inds = nd_arr == 'act'
    rest_inds = nd_arr == 'rest'
    tens[act_inds, 1] = 1
    tens[rest_inds, 0] = 1
    return tens

def expand(arr):
    tens = torch.zeros(arr.shape)
    tens[np.arange(arr.shape[0]), torch.argmax(arr, dim=1)] = 1
    return tens

model_type = "CellClassifier"
num_classes = 2
model_directory = "/u/scratch/s/schhina/temp_geneformer_output/240509125208/240509_geneformer_cellClassifier_cm_classifier_test/ksplit1"
n_samples = 100

model = pu.load_model(model_type, num_classes, model_directory, mode="eval")
# model.train()

input_id_tensor = torch.load("/u/scratch/s/schhina/temp_files/input_ids.pt")
atten_mask_tensor = torch.load("/u/scratch/s/schhina/temp_files/atten_mask.pt")
labels_tensor = torch.load("/u/scratch/s/schhina/temp_files/labels.pt")
lens_tensor = torch.load("/u/scratch/s/schhina/temp_files/lens.pt")
downsampled_data_tensor = torch.load("/u/scratch/s/schhina/temp_files/downsampled_data.pt")

output = model(input_ids=input_id_tensor[:n_samples].to("cuda"), attention_mask=atten_mask_tensor[:n_samples],)
output.hidden_states[-1].retain_grad()

# mean_embs = mean_nonpadding_embs(output.hidden_states[-1], lens_tensor[:n_samples])
# mean_embs.retain_grad()
# print(mean_embs.shape)

model_input_size = pu.get_model_input_size(model)
token_dictionary_file = "/u/home/s/schhina/.conda/envs/geneformer_env/lib/python3.11/site-packages/geneformer/token_dictionary.pkl"
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)

token_gene_dict = {v: k for k, v in gene_token_dict.items()}
# print(token_gene_dict)
pad_token_id = gene_token_dict.get("<pad>")

y = one_hot(labels_tensor)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output.logits, y[:n_samples].to("cuda"))
loss.backward()

# for param in model.parameters():
#     print(param.grad.shape)

# print(mean_embs)
# mean_embs.backward()
# print(mean_embs.grad)

embs_stack = pu.pad_tensor_list(
        [output.hidden_states[-1].grad],
        2048, # overall_max_len
        pad_token_id,
        model_input_size, # pu.get_model_input_size(model)
        1, # dim
        pu.pad_3d_tensor, 
    )

# print(embs_stack)
# print(embs_stack.shape)

embd_df = ee.label_gene_embs(embs_stack, downsampled_data_tensor, token_gene_dict)

hidden_grad = output.hidden_states[-1].grad
print(hidden_grad[0])
print(hidden_grad.shape)

print(embd_df.head())
print(embd_df.shape)

l2_norms = np.linalg.norm(embd_df, axis=1, ord=1)
gene_norms = dict(zip(embd_df.index, l2_norms))
print(gene_norms)
for k, v in gene_norms.items():
    if v != 0:
        print((k, v))

norms = torch.linalg.norm(hidden_grad, dim=2, ord=2)
# print(norms)
# print(norms.shape)

# print(output.logits.grad)
# print(input_id_tensor.grad)
