# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:42:08 2021

@author: fengshuai
"""
import numpy as np
import yaml  #yaml是一个专门用来写配置文件的语言,因其简洁高效而被大众喜爱。

# xxx.yaml 配置文件中的数值型参数一般是以键值对形式(字典)存在：

def load_yaml(path:str):
    with open(path,"r") as file:
        try:
            yaml_file = yaml.safe_load(file) # 加载配置文件中的所有内容，yaml_file返回的是一个python字典：例如：{'gama': 0.001, 'sigma': 8.5}形式
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_file