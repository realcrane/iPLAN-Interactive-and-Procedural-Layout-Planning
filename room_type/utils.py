from collections import OrderedDict

room_label = [(0,'LivingRoom'),
            (1,'MasterRoom'),
            (2,'Kitchen'),
            (3,'Bathroom'),
            (4,'DiningRoom'),
            (5,'ChildRoom'),
            (6,'StudyRoom'),
            (7,'SecondRoom'),
            (8,'GuestRoom'),
            (9,'Balcony'),
            (10,'Entrance'),
            (11,'Storage'),
            (12,'Wall-in'),
            (13,'External'),
            (14,'ExteriorWall'),
            (15,'FrontDoor'),
            (16,'Interior')]

total_num_category = len(room_label)

def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()

class Loss_AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, loss_name):
        self.loss_names = loss_name
        self.reset()

    def reset(self):
        """Reset the loss value"""
        self.avg = OrderedDict()
        self.sum = OrderedDict()
        self.val = OrderedDict()
        self.count = 0

        for name in self.loss_names:
            self.avg[name] = 0.0
            self.sum[name] = 0.0
            self.val[name] = 0.0

    def update(self, losses, n):
        self.count += n
        for name, value in zip(self.loss_names, losses):
            self.val[name] = value / n
            self.sum[name] += value
            self.avg[name] = self.sum[name] / self.count