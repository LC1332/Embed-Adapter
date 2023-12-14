"""
@Desc: 全局配置文件读取
"""
import argparse
import yaml
from typing import Dict, List
import os
import shutil
import sys
DEFAULT_CONFIG_PATH = "../default_config.yml"
class DynamicConfig:
    def __init__(self, config_data: dict, parent_key: str = ""):
        for key, value in config_data.items():
            if isinstance(value, dict):
                # 递归创建子配置对象
                setattr(self, key , DynamicConfig(value, key))
            else:
                # 直接设置属性
                setattr(self, key, value)
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"Config item '{item}' not found")
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items()})
    

class Config:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path) and os.path.isfile(DEFAULT_CONFIG_PATH):
            shutil.copy(src=DEFAULT_CONFIG_PATH ,dst=config_path)
            print(
                f"已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。"
            )
            sys.exit(0)
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            self.mirror: str = yaml_config["mirror"]
            # TODO: 自己的预处理配置
            self.corr_map = DynamicConfig(yaml_config["corr_map"])


parser = argparse.ArgumentParser()
# 为避免与以前的config.json起冲突，将其更名如下
parser.add_argument("-y", "--yml_config", type=str, default="../config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)


