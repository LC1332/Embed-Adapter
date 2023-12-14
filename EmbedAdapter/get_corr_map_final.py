import json
import math
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

# 配置参数
BATCH_SIZE = 400
GENERAL_BATCH_SIZE = 16
MODEL_NAME_BGE_SMALL = "BAAI/bge-small-zh-v1.5"
MODEL_NAME_BGE_BASE = "BAAI/bge-base-zh-v1.5"
FILE_NAME = "your_file_name_here.json"
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

openai_config = {
    "name":"openai",
    "long_name":"openai",
    "embeddings":openai_embeddings, # 预先抽取的
    "batch_embed_fun":None
}
bge_small_config = {
    "name":"bge_small_zh_15",
    "long_name":"BAAI/bge-small-zh-v1.5",
    "embeddings":None,
    "batch_embed_fun":bge_small_zh_fun
}
bge_base_config = {
    "name":"bge_base_zh_15",
    "long_name":"BAAI/bge-base-zh-v1.5",
    "embeddings":None,
    "batch_embed_fun":bge_base_zh_fun
}


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


def bge_small_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-small-zh-v1.5", return_tensor = True)

def bge_base_zh_fun( texts ):
    return get_general_embeddings(texts, "BAAI/bge-base-zh-v1.5", return_tensor = True)

def compute_correlations(datas, method_configs):
    """ 计算不同方法之间的相关性 """
    texts = [data["text"] for data in datas]
    n = len(texts)
    n_methods = len(method_configs)
    corr_map = {}

    for start_id in tqdm(range(0, n, BATCH_SIZE)):
        end_id = min(start_id + BATCH_SIZE, n)
        texts_batch = texts[start_id:end_id]

        method2embeddings = {}
        for config in method_configs:
            if config["embeddings"] is None:
                embeddings = batch_process_embeddings(texts_batch, config["model_name"])
            else:
                embeddings = config["embeddings"][start_id:end_id]
            method2embeddings[config["name"]] = embeddings

        # 略去具体的计算过程 ...

    return corr_map

def main():
    
    datas = load_data(FILE_NAME)
    texts = [data["text"] for data in datas]
    openai_embeddings = [data["embedding"] for data in datas]
    method_configs = [openai_config, bge_small_config, bge_base_config]
    forbidden_pairs = []  # 如果你某个pair不希望生成，需要把 model_A_2_model_B 放到这个list里面
    n = len(texts)

    n_method = len(method_configs)
    corr_map = compute_correlations(datas, method_configs)



    # 进一步处理 corr_map ...

if __name__ == "__main__":
    main()
