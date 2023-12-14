"""
@Desc: 全局配置文件读取
"""
import argparse
import yaml
from typing import Dict, List
import os
import shutil
import sys

class DynamicConfig:
    def __init__(self, config_data: dict, parent_key: str = ""):
        for key, value in config_data.items():
            if isinstance(value, dict):
                # 递归创建子配置对象
                setattr(self, key + "_config", DynamicConfig(value, key))
            else:
                # 直接设置属性
                setattr(self, key, value)

    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items()})
    

class Config:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            print(
                f"已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。"
            )
            print("如无特殊需求，请勿修改default_config.yml或备份该文件。")
            sys.exit(0)
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            self.mirros: str = yaml_config["mirros"]
            # TODO: 自己的预处理配置
            self.mypreprocess_config = DynamicConfig(yaml_config["mypreprocess"])


parser = argparse.ArgumentParser()
# 为避免与以前的config.json起冲突，将其更名如下
parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)