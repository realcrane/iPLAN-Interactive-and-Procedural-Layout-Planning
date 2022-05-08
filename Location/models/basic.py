from collections import OrderedDict
import torch as t
import os
import re

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.name = str(type(self))

    def load_model(self, path):
        self.load_state_dict(t.load(path))

    def save_model(self, epoch=0):  
        pth_list = [pth for pth in os.listdir('checkpoints') if re.match(self.name, pth)]
        pth_list = sorted(pth_list, key=lambda x: os.path.getmtime(os.path.join('checkpoints', x)))
        if len(pth_list) >= 10 and pth_list is not None:
            to_delete = 'checkpoints/' + pth_list[0]
            if os.path.exists(to_delete):
                os.remove(to_delete)  
                
        path = f'checkpoints/{self.name}_{epoch}.pth'
        t.save(self.state_dict(), path)