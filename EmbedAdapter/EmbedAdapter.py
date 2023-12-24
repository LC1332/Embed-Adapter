from .model_name_mapping import MODEL_NAME_MAPPING
import numpy as np

class EmbedAdapter:
    def __init__(self, source, output, implementation="hf"):
        self.source_long_name = MODEL_NAME_MAPPING.get(source, source)
        self.output_long_name = MODEL_NAME_MAPPING.get(output, output)
        self.implementation = implementation
        self.adapter = self._select_adapter()

    def _select_adapter(self):
        if self.implementation == "naive":
            from .NaiveAdapter import NaiveAdapter
            return NaiveAdapter(self.source_long_name, self.output_long_name)
        elif self.implementation == "hf":
            from .HFAdapter import HFAdapter
            return HFAdapter(self.source_long_name, self.output_long_name)
        # 可以添加更多的实现
        elif self.implementation == "combine":
            from .CombineAdapter import CombineAdapter
            return CombineAdapter(self.source_long_name, self.output_long_name)
        else:
            raise ValueError("Unknown implementation")

    def __call__(self, input_embedding, output_format="list"):
        return self.adapter.convert(input_embedding, output_format)

    # 可能还需要其他辅助方法
