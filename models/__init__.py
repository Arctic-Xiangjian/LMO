# -- coding: utf-8 --
from importlib import import_module
from utils import networks
from .patch_merge_module.import_module import PatchMergeModule

class Model:
    """
    Used to invoke different network structures in the models module
    """
    def __init__(self, args):
        print('==> Making model......')

        module = import_module('models.' + args.model.lower())
        self.model = module.make_model(args)

    def __repr__(self):
        return "This is the class that calls the different models in the models module."


# if __name__ == '__main__':
#     from option import args
#
#     print(Model(args))
