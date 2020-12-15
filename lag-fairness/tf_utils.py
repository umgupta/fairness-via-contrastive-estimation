import tensorflow as tf
import logging

class  Logger:
    """dummy logger class to make things work"""

    def __init__(self):
        pass
    @staticmethod
    def configure(*args, **kwargs):
        pass
    @staticmethod
    def log(string, *args, **kwargs):
        print(string)
    @staticmethod
    def dumpkvs(*args, **kwargs):
        pass
    @staticmethod
    def logkvs(*args, **kwargs):
        pass

def gpu_session(*args, **kwargs):
    # just using tf cpu so return default session object
    return tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=10))

def clear_dir(*args, **kwargs):
    pass

def find_available_gpu(*args, **kwargs):
    pass
logger = Logger

class Datasets:
    def __init__(self, train, test):
        self.train = train
        self.test  = test


def obtain_log_path(str):
    return "result/"+str