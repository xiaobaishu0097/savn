""" Contains the Episodes for Navigation. """
import random

import torch
import os

import numpy as np

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY
from datasets.constants import DONE
from datasets.environment import Environment
from datasets.glove import Glove

from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode

CLASSES = [
    'Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl',
    'Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl',
    'DeskLamp', 'CellPhone', 'Book', 'AlarmClock',
    'Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch'
]

class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.det_frame = None

        self.last_det = False
        self.current_det = False
        self.det_gt = None
        self.optimal_actions=None

        self.scene_states = []
        self.detections = []
        if args.eval:
            random.seed(args.seed)

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int, arrive):

        self.last_det = self.current_det
        action = self.actions_list[action_as_int]

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful, arrive = self.judge(action, arrive)
        return reward, terminal, action_was_successful, arrive

    def judge(self, action, arrive):
        """ Judge the last event. """
        reward = STEP_PENALTY

        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    break
        else:
            # test for 100% accuracy of target detection
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    arrive = True
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    break
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful, arrive

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def _new_episode(
        self, args, scenes, possible_targets, targets=None, keep_obj=False, optimal_act=None, glove=None, det_gt=None,
    ):
        """ New navigation episode. """
        scene = random.choice(scenes)

        img_file_scene = args.images_file_name

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                # images_file_name=args.images_file_name,
                images_file_name=img_file_scene,
                local_executable_path=args.local_executable_path,
                total_images_file=None
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # Randomize the start location.
        self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

        # glove = Glove(os.path.join(args.glove_dir, self.environment.controller.scene_name, 'det_feature.hdf5'))
        glove = glove[self.environment.controller.scene_name]
        if optimal_act is not None:
            self.optimal_actions = optimal_act[self.environment.controller.scene_name][self.task_data[0]]
        else:
            self.optimal_actions = None

        self.glove_embedding = None

        init_pos = '{}|{}|{}|{}'.format(
            # self.environment.controller.scene_name,
            self.environment.controller.state.position()['x'],
            self.environment.controller.state.position()['z'],
            self.environment.controller.state.rotation,
            self.environment.controller.state.horizon
        )

        target_embedding_array = np.zeros((len(CLASSES), 1))
        target_embedding_array[CLASSES.index(self.target_object)] = 1
        # glove_embedding_tensor = np.concatenate((glove.glove_embeddings[init_pos][()], target_embedding_array), axis=1)
        glove_embedding_tensor = np.concatenate((glove[init_pos], target_embedding_array), axis=1)

        self.glove_embedding = toFloatTensor(
            glove_embedding_tensor, self.gpu_id
        )
        # self.glove_reader = glove.glove_embeddings
        self.glove_reader = glove
        # self.det_gt = det_gt[self.environment.controller.scene_name]

        # self.glove_embedding = toFloatTensor(
        #     glove.glove_embeddings[goal_object_type][:], self.gpu_id
        # )

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        optimal_act=None,
        glove=None,
        # det_gt=None
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        # self.last_det = False
        # self.current_det = False
        self.det_frame = None
        self.detections = []
        self._new_episode(args, scenes, possible_targets, targets, keep_obj, optimal_act=optimal_act, glove=glove)
