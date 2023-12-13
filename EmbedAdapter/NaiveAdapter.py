import numpy as np

_pseudo_inverses =  None

class NaiveAdapter:
    def __init__(self, source_long_name, target_long_name):
        self.source_long_name = source_long_name
        self.target_long_name = target_long_name
        # 初始化任何必要的资源

        self.pesudo_inverse = self.get_pesudo_inverse(source_long_name, target_long_name)

    def get_pesudo_inverse( self,source_long_name, target_long_name ):
        self.__download_pesudo_inverse()
        global _pseudo_inverses
        pesudo_key = (source_long_name, target_long_name)
        if pesudo_key in _pseudo_inverses:
            return _pseudo_inverses[pesudo_key]
        else:
            print('pesudo inverse not found!', pesudo_key)
            return None

    def __download_pesudo_inverse( self ):
        global _pseudo_inverses
        if _pseudo_inverses is None:
            print('now download pseudo inverse')

            import os
            # if not exist the file pseudo_inverses.pkl
            # download from somelink.pkl
            if not os.path.exists('pseudo_inverses.pkl'):
                import urllib.request

                # Download pseudo_inverses.pkl from somelink.pkl
                url = 'https://github.com/LC1332/Embed-Adapter/raw/main/data/pseudo_inverses.pkl'
                urllib.request.urlretrieve(url, 'pseudo_inverses.pkl')

            # Load pseudo_inverses from pseudo_inverses.pkl
            import pickle
            with open('pseudo_inverses.pkl', 'rb') as file:
                _pseudo_inverses = pickle.load(file)

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
