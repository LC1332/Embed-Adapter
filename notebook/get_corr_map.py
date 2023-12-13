fname = "../data/zhwiki_2k_embedding.jsonl"

import json

raw_datas = []

with open(fname, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() == "":
            continue
        data = json.loads(line)
        raw_datas.append(data)

datas = []
for data in raw_datas:
    datas.append({
        "text": data["left"],
        "embedding": data["left_embedding"],
    })
    datas.append({
        "text": data["right"],
        "embedding": data["right_embedding"],
    })
print(len(datas))
# datas = datas[:16]
from tqdm import tqdm
import torch

#TODO 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_pool = {}
_tokenizer_pool = {}

# BAAI/bge-small-zh-v1.5
def get_general_embeddings( sentences , model_name = "BAAI/bge-small-zh-v1.5" , return_tensor = False):

    global _model_pool
    global _tokenizer_pool

    if model_name not in _model_pool:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer_pool[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_pool[model_name] = AutoModel.from_pretrained(model_name).to(device)

    _model_pool[model_name].eval()

    # Tokenize sentences
    encoded_input = _tokenizer_pool[model_name](sentences, padding=True, truncation=True, return_tensors='pt', max_length = 512).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = _model_pool[model_name](**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]

    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    if return_tensor == True:
        return sentence_embeddings
    return sentence_embeddings.cpu().tolist()

def get_general_embedding( text_or_texts , model_name = "BAAI/bge-small-zh-v1.5" ):
    if isinstance(text_or_texts, str):
        return get_general_embeddings([text_or_texts], model_name)[0]
    else:
        return get_general_embeddings_safe(text_or_texts, model_name)

general_batch_size = 16

import math

def get_general_embeddings_safe(sentences, model_name = "BAAI/bge-small-zh-v1.5"):

    embeddings = []

    num_batches = math.ceil(len(sentences) / general_batch_size)

    for i in tqdm( range(num_batches) ):
        # print("run bge with batch ", i)
        start_index = i * general_batch_size
        end_index = min(len(sentences), start_index + general_batch_size)
        batch = sentences[start_index:end_index]
        embs = get_general_embeddings(batch, model_name)
        embeddings.extend(embs)

    return embeddings

def get_bge_zh_embedding( text_or_texts):
    return get_general_embedding(text_or_texts, "BAAI/bge-small-zh-v1.5")

#
# test_batch_n = 8
# test_batch = [data["text"] for data in datas[:test_batch_n]]
# bge_small_15 = get_general_embeddings(test_batch, "BAAI/bge-small-zh-v1.5", return_tensor = True)

texts = [data["text"] for data in datas]
openai_embeddings = [data["embedding"] for data in datas]

openai_config = {
    "name":"openai",
    "long_name":"openai",
    "embeddings":openai_embeddings, # 预先抽取的
    "batch_embed_fun":None
}

def bge_small_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-small-zh-v1.5", return_tensor = True)

bge_small_config = {
    "name":"bge_small_zh_15",
    "long_name":"BAAI/bge-small-zh-v1.5",
    "embeddings":None,
    "batch_embed_fun":bge_small_zh_fun
}

def bge_base_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-base-zh-v1.5", return_tensor = True)

bge_base_config = {
    "name":"bge_base_zh_15",
    "long_name":"BAAI/bge-base-zh-v1.5",
    "embeddings":None,
    "batch_embed_fun":bge_base_zh_fun
}

method_configs = [openai_config, bge_small_config, bge_base_config]

forbidden_pairs = [] # 如果你某个pair不希望生成，需要把 model_A_2_model_B 放到这个list里面

n = len(texts)

n_method = len(method_configs)
batch_size = 400

from tqdm import tqdm

corr_map = {}

for start_id in tqdm(range(0,n, batch_size)):
    end_id = min(start_id + batch_size, n)
    texts_batch = texts[start_id:end_id]

    method2embeddings = {}

    for method_config in method_configs:
        method_name = method_config["name"]
        if method_config["embeddings"] is None:
            embeddings = method_config["batch_embed_fun"](texts_batch)
        else:
            embeddings = torch.tensor(method_config["embeddings"][start_id:end_id]).to(device)


        method2embeddings[method_name] = embeddings

    for method_i in range(n_method):
        for method_j in range(method_i, n_method):
            corr_index = ( method_i, method_j )
            X_1 = method2embeddings[method_configs[method_i]["name"]]
            X_2 = method2embeddings[method_configs[method_j]["name"]]
            X1TX2 = X_1.T @ X_2
            if corr_index in corr_map:
                corr_map[corr_index] += X1TX2
            else:
                corr_map[corr_index] = X1TX2

for method_i in range(n_method - 1):
    for method_j in range(method_i + 1, n_method):
        corr_index = ( method_i, method_j )
        trans_index = (method_j, method_i )
        corr_map[trans_index] = corr_map[corr_index].T

for it in corr_map.keys():
    corr_map[it] = corr_map[it].cpu()

import numpy as np
print('开始计算伪逆')
pseudo_inverses = {}
i,i=0,0
for i in range(n_method):
    # 计算每个方法自身相关性矩阵的伪逆
    corr_ii = corr_map[(i, i)].numpy()
    pseudo_inv_ii = np.linalg.pinv(corr_ii)

    for j in range(n_method):
        if i != j:
            # 计算交叉相关性矩阵
            corr_ij = corr_map[(i, j)].numpy()
            # 计算伪逆
            pseudo_inverse_ij = pseudo_inv_ii @ corr_ij
            pseudo_key = (method_configs[i]["long_name"] , method_configs[j]["long_name"])
            pseudo_inverses[pseudo_key] = pseudo_inverse_ij



print('保存伪逆')

import pickle
# Save corr_map to a pickle file
with open('../data/pseudo_inverses_final.pkl', 'wb') as f:
    pickle.dump(pseudo_inverses, f)


#TODO torch版本计算伪逆
# for i in range(n_method):
#     corr_ii = corr_map[(i, i)]
#     # 计算伪逆，使用PyTorch的pinverse函数
#     pseudo_inv_ii = torch.pinverse(corr_ii)
#
#     for j in range(n_method):
#         if i != j:
#             # 同样，从GPU获取Tensor
#             corr_ij = corr_map[(i, j)]
#             # 使用PyTorch进行矩阵乘法
#             pseudo_inverse_ij = torch.matmul(pseudo_inv_ii, corr_ij)
#             pseudo_key = (method_configs[i]["long_name"], method_configs[j]["long_name"])
#             # 这里不清楚后面是不是需要np数组，所以为了兼容性 转成np再保存
#             pseudo_inverses[pseudo_key] = pseudo_inverse_ij.cpu().numpy()









