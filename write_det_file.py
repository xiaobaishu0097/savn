import datetime
import multiprocessing as mp
import numpy
import ai2thor
import os
import h5py
import re

OBJECT_DICT = [
    ['toaster', 'microwave', 'fridge', 'coffee maker', 'garbage can', 'box', 'bowl'],
    ['pillow', 'laptop', 'television', 'garbage can', 'box', 'bowl'],
    ['plant', 'lamp', 'book', 'alarm clock'],
    ['sink', 'toilet paper', 'soap bottle', 'light switch']
]

def write_det(scene):

    # judge the type of the scene and get the possible object typies list
    scene_num = int(int(re.findall('\d+', scene)[0]) / 100)
    object_list = OBJECT_DICT[scene_num]

    # read the possible position list and get the hdf5 file path to store the detection data
    data_dir = './data/thor_offline_data/'
    scene_dir = os.path.join(data_dir, scene)

    det_hdf5_path = os.path.join(scene_dir, 'env_det.hdf5')

    loc_file = os.path.join(scene_dir, 'resnet18_featuremap.hdf5')
    loc_data = h5py.File(loc_file, 'r')
    loc_data = loc_data.keys()

    # initialize the controller to get the enviroment feedback
    controller = ai2thor.controller.Controller()
    controller.start()
    controller.reset(scene)
    event = controller.step(dict(action='Initialize', gridSize=0.25, renderObjectImage=True))

    # default z as 1
    z = 1

    scene_det = {}

    for loc in loc_data:
        x, y, rot, hor = loc.split('|')
        event = controller.step(dict(action='TeleportFull', x=x, y=y, z=z, rotation=rot, horizon=hor))

        det_data = event.class_detections2D

        det_loc = {}

        for det in det_data:
            for obj in object_list:
                if obj in det:
                    if obj not in det_loc:
                        det_loc[obj] = []

                    for i in range(len(det_data[det])):
                        det_loc[obj].append(det_data[det][i])

        scene_det[loc] = det_loc

    with h5py.File(det_hdf5_path, 'w') as dh:
        for key, value in scene_det.items():
            dh.create_dataset(key, data=value)

    print('{} has complete!'.format(scene))



if __name__ == '__main__':
    print(datetime.datetime.now())

    scene_dir = './data/thor_offline_data/'
    scene_list = os.listdir(scene_dir)

    with mp.Pool(processes=10) as pool:
        for num, scene in enumerate(scene_list):
            pool.map(write_det, scene)

            if num % 10 == 0:
                print('{} dirs have started to process')

    print('All dirs completed!')
    print(datetime.datetime.now())



