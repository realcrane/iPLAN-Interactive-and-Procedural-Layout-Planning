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

category = [category for category in room_label if category[1] not in set(['External',\
            'ExteriorWall', 'FrontDoor', 'Interior'])]

num_category = len(category)

pixel2length = 18/256

def label2name(label=0):
    if label < 0 or label > 16:
        raise Exception("Invalid label!", label)
    else:
        return room_label[label][1]

def label2index(label=0):
    if label < 0 or label > 16:
        raise Exception("Invalid label!", label)
    else:
        return label

def index2label(index=0):
    if index < 0 or index > 16:
        raise Exception("Invalid index!", index)
    else:
        return index

def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()