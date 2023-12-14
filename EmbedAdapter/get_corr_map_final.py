import json
import pickle
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from tqdm import tqdm

from config import config

# 配置参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型和分词器池
_model_pool = {}
_tokenizer_pool = {}

def load_data(fname):
    """ 加载数据 """
    datas = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                datas.extend([
                    {"text": data["left"], "embedding": data["left_embedding"]},
                    {"text": data["right"], "embedding": data["right_embedding"]}
                ])
    return datas

# BAAI/bge-small-zh-v1.5
def get_general_embeddings( sentences , model_name = "BAAI/bge-small-zh-v1.5" , return_tensor = False):

    global _model_pool
    global _tokenizer_pool

    if model_name not in _model_pool:

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


def bge_small_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-small-zh-v1.5", return_tensor = True)

def bge_base_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-base-zh-v1.5", return_tensor = True)

def compute_correlations(datas, method_configs, n, batch_size, n_method, texts):
    """ 计算不同方法之间的相关性 """
    corr_map = {}
    for start_id in tqdm(range(0, n, batch_size)):
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
                corr_index = (method_i, method_j)
                X_1 = method2embeddings[method_configs[method_i]["name"]]
                X_2 = method2embeddings[method_configs[method_j]["name"]]
                X1TX2 = X_1.T @ X_2
                if corr_index in corr_map:
                    corr_map[corr_index] += X1TX2
                else:
                    corr_map[corr_index] = X1TX2

    for method_i in range(n_method - 1):
        for method_j in range(method_i + 1, n_method):
            corr_index = (method_i, method_j)
            trans_index = (method_j, method_i)
            corr_map[trans_index] = corr_map[corr_index].T
    return corr_map


def compute_pseudo_inverses(datas, method_configs, n_method, corr_map):
    # TODO 转到cpu上
    for it in corr_map.keys():
        corr_map[it] = corr_map[it].cpu()
    pseudo_inverses={}
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
                pseudo_key = (method_configs[i]["long_name"], method_configs[j]["long_name"])
                pseudo_inverses[pseudo_key] = pseudo_inverse_ij



def save(path,data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
def main():
    fname = config.corr_map.fname
    batch_size = config.corr_map.batch_size
    openai_config = config.openai_config
    bge_small_config = config.bge_small_config
    bge_base_config = config.bge_base_config

    datas = load_data(fname)
    texts = [data["text"] for data in datas]
    openai_embeddings = [data["embedding"] for data in datas]
    method_configs = [openai_config, bge_small_config, bge_base_config]
    forbidden_pairs = []  # 如果你某个pair不希望生成，需要把 model_A_2_model_B 放到这个list里面
    n = len(texts)
    batch_size=1
    n_method = len(method_configs)
    corr_map = compute_correlations(datas, method_configs, n, batch_size, n_method, texts)
    pseudo_inverses=compute_pseudo_inverses(datas, method_configs, n_method, corr_map)
    print(pseudo_inverses)
    # Save corr_map to a pickle file
    # save('../data/pseudo_inverses_final.pkl',pseudo_inverses)

if __name__ == "__main__":





    te.bge_small_config['name']
    te.batch_embed_fun("1")
    te
    print()





    # main()
    print()
