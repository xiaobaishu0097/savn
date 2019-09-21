""" Contains the Episodes for Navigation. """
from datasets.environment import Environment
from utils.net_util import gpuify, toFloatTensor
from .basic_episode import BasicEpisode
import pickle
from datasets.data import num_to_name
import numpy as np
from datasets.offline_controller_with_small_rotation import ThorAgentState

CLASSES = [
    'Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl',
    'Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl',
    'DeskLamp', 'CellPhone', 'Book', 'AlarmClock',
    'Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch'
]

class TestValEpisode(BasicEpisode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(TestValEpisode, self).__init__(args, gpu_id, strict_done)
        self.file = None
        self.all_data = None
        self.all_data_enumerator = 0

    def _new_episode(self, args, episode, glove):
        """ New navigation episode. """
        scene = episode["scene"]

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # self.environment.controller.state = episode["state"]
        y = 0.9009995
        x, z, hor, rot = episode["state"].split('|')
        self.environment.controller.state = ThorAgentState(float(x), float(y), float(z), float(hor), float(rot))

        self.task_data = episode["task_data"]
        self.target_object = episode["goal_object_type"]

        if args.verbose:
            print("Scene", scene, "Navigating towards:", self.target_object)

        # self.glove_embedding = gpuify(episode["glove_embedding"], self.gpu_id)
        glove = glove[self.environment.controller.scene_name]

        self.glove_embedding = None

        init_pos = '{}|{}|{}|{}'.format(
            # self.environment.controller.scene_name,
            self.environment.controller.state.position()['x'],
            self.environment.controller.state.position()['z'],
            self.environment.controller.state.rotation,
            self.environment.controller.state.horizon
        )
        # init_pos = self.environment.controller.state

        target_embedding_array = np.zeros((len(CLASSES), 1))
        target_embedding_array[CLASSES.index(self.target_object)] = 1
        glove_embedding_tensor = np.concatenate((glove[init_pos], target_embedding_array), axis=1)

        self.glove_embedding = toFloatTensor(
            glove_embedding_tensor, self.gpu_id
        )
        self.glove_reader = glove

        return True

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        glove=None,
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None

        if self.file is None:
            sample_scene = scenes[0]
            if "physics" in sample_scene:
                scene_num = sample_scene[len("FloorPlan") : -len("_physics")]
            else:
                scene_num = sample_scene[len("FloorPlan") :]
            scene_num = int(scene_num)
            scene_type = num_to_name(scene_num)
            task_type = args.test_or_val
            self.file = open(
                "test_val_split/" + scene_type + "_" + task_type + ".pkl", "rb"
            )
            self.all_data = pickle.load(self.file)
            self.file.close()
            self.all_data_enumerator = 0

        episode = self.all_data[self.all_data_enumerator]
        self.all_data_enumerator += 1
        self._new_episode(args, episode, glove)
