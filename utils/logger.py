# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:55:58 2021

@author: fengshuai
"""
import os
import os.path as osp
from pathlib import Path
import sys
import errno



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno!= errno.EEXIST:  #errno.EEXIST 文件已存在　
                raise  #raise：单独一个 raise。该语句引发当前上下文中捕获的异常（比如在 except 块中），或默认引发 RuntimeError 异常

class Logger:
    """ Write console output to external text file.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_
    Args:
        fpath (str): directory to save logging file.
    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """
    
    def __init__(self,fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath,"w")
            
    def __del__(self):
        self.close()
        
    def __enter__(self):
        pass
    
    def __exit__(self,*args):
        self.close()
        
    def write(self,msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
            
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
    


def setup_logger(output_dir,output_name): # setup_logger(str(output_dir))
   # create dir
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True,exist_ok=True)
    
    # create file
    file_path = dir_path/output_name
    file_path.touch(exist_ok=True)
    
    sys.stdout = Logger(file_path)    
    # sys.stdout就像是一个类文件对象,可以将它赋值给任意的一个文件对象,重定向输出
    # 原始的sys.stdout指向控制台，如果把文件的对象引用赋给sys.stdout，那么print调用的就是文件对象的write方法