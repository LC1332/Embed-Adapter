
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
def load_matrix( dataset ):
    n_row = len(dataset["train"])
    n_col = len(dataset["train"][0])
    matrix = np.zeros((n_row,n_col), dtype = float )
    for i in tqdm(range(n_row)):
        matrix[i] = list( dataset["train"][i].values()) 
    return matrix


class HFAdapter:
    def __init__(self, source_long_name, target_long_name, model_name = None):
        self.source_long_name = source_long_name
        self.target_long_name = target_long_name
        # 初始化任何必要的资源
        self.pesudo_inverse = self.get_pesudo_inverse(source_long_name, target_long_name, model_name)


    def get_pesudo_inverse( self,source_long_name, target_long_name , model_name = None):
        if model_name == None:
            str1 = source_long_name.replace("/","---")
            str2 = target_long_name.replace("/","---")
            model_name = str1 + "__" + str2 + ".parquet"
            dataset = load_dataset("silk-road/Embedding-Adapter",data_files={'train':model_name})
        else:
            print("try download adapte model from ", model_name , " but not yet implemented loading code")

        return load_matrix(dataset)

    def convert(self, input_embedding, output_format):
        # 这里实现转换逻辑
        if isinstance(input_embedding, list):
            input_embedding = np.array(input_embedding)

        # input_embedding is 1 * input_dim np array
        # self.pesudo_inverse is input_dim * output_dim np array
        # output_embedding is 1 * output_dim np array
        output_embedding = np.matmul(input_embedding, self.pesudo_inverse)

        if output_format == "np":
            return output_embedding
        elif output_format == "list":
            return output_embedding.tolist()
        else:
            raise ValueError("Unsupported output format")
