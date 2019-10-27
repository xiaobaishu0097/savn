import torch
import numpy as np

from utils.net_util import gpuify, toFloatTensor
from models.model_io import ModelInput
from runners.train_util import generate_det_4_iou

from .agent import ThorAgent

CLASSES = [
    'Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl',
    'Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl',
    'DeskLamp', 'CellPhone', 'Book', 'AlarmClock',
    'Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch'
]

class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.detector_det = None
        self.gt_det = None
        self.optim_step = None
        self.last_det = None

    def eval_at_state(self, model_options):
        model_input = ModelInput()
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame
        model_input.hidden = self.hidden

        current_pos = '{}|{}|{}|{}'.format(
            # self.environment.scene_name,
            self.episode.environment.controller.state.position()['x'],
            self.episode.environment.controller.state.position()['z'],
            self.episode.environment.controller.state.rotation,
            self.episode.environment.controller.state.horizon
        )

        target_embedding_array = np.zeros((len(CLASSES), 1))
        target_embedding_array[CLASSES.index(self.episode.target_object)] = 1
        glove_embedding_tensor = np.concatenate((self.episode.glove_reader[current_pos][()], target_embedding_array), axis=1)
        # model_input.target_class_embedding = self.episode.glove_embedding
        # if ((self.episode.glove_reader[current_pos][CLASSES.index(self.episode.target_object)] != np.array([0, 0, 0, 0])).all()) and (self.episode.det_frame == None):
        #     self.episode.current_det = self.eps_len
        model_input.target_class_embedding = toFloatTensor(glove_embedding_tensor, self.gpu_id)

        # if self.eps_len == 0:
        #     model_input.det_his = toFloatTensor(torch.zeros(4), self.gpu_id)
        # else:
        #     model_input.det_his = self.last_det
        # model_input.det_cur = toFloatTensor(self.episode.glove_reader[current_pos][CLASSES.index(self.episode.target_object)], self.gpu_id)
        # self.last_det = model_input.det_cur

        model_input.action_probs = self.last_action_probs

        # if self.eps_len == 0:
        #     det_iou = toFloatTensor(torch.zeros(1, 4), self.gpu_id)
        # else:
        #     det_iou = generate_det_4_iou(self.last_det)
        #     det_iou = toFloatTensor(det_iou, self.gpu_id)
        #
        # model_input.action_probs = torch.cat((self.last_action_probs, det_iou), dim=1)

        self.episode.detections.append(self.episode.glove_reader[current_pos][CLASSES.index(self.episode.target_object)])
        # self.detector_det = self.episode.glove_reader[current_pos][CLASSES.index(self.episode.target_object)]
        # if self.episode.task_data[0] in self.episode.det_gt[current_pos]:
        #     self.gt_det = self.episode.det_gt[current_pos][self.episode.task_data[0]]
        # else:
        #     self.gt_det = np.zeros(4)
        # optimal_solution = self.episode.environment.controller.shortest_path_to_target(current_pos, self.episode.task_data[0])
        if self.episode.optimal_actions is not None:
            optimal_solution = self.episode.optimal_actions[current_pos]
            self.optim_step = torch.zeros((1, 6))
            self.optim_step[0, optimal_solution] = 1
        # if (optimal_solution[0] is not None) and (len(optimal_solution[0]) < 99):
        #     optim_next_state = optimal_solution[0][0]
        # else:
        #     optim_next_state = current_pos
        # self.optim_step = self.optim_action(current_pos, optim_next_state)

        # optim_steps = torch.zeros((1, 1))
        # optim_steps[0, 0] = self.episode.environment.controller.shortest_path_to_target(current_pos, self.episode.task_data[0])[1]
        # model_input.optim_steps = toFloatTensor(optim_steps, self.gpu_id)

        # det_his = torch.zeros((1, 2))
        # if self.episode.current_det:
        #     det_his[0, 1] = 1
        # if self.episode.last_det:
        #     det_his[0, 0] = 1
        # model_input.det_relation = toFloatTensor(det_his, self.gpu_id)

        return model_input, self.model.forward(model_input, model_options)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                )
        else:
            self.hidden = (
                torch.zeros(1, self.hidden_state_sz),
                torch.zeros(1, self.hidden_state_sz),
            )
        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def optim_action(self, current_state, next_state):
        action_prob = torch.zeros((1, 6))
        if current_state == next_state:
            action_prob[0, 5] = 1
        else:
            current_x, current_z, current_rot, current_hor = current_state.split('|')
            next_x, next_z, next_rot, next_hor = next_state.split('|')
            if int(next_rot) == (int(current_rot) + 45):
                action_prob[0, 2] = 1
            elif int(next_rot) == (int(current_rot) - 45):
                action_prob[0, 1] = 1
            elif (current_hor == '0') and (next_hor == '30'):
                action_prob[0, 4] = 1
            elif (current_hor == '30') and (next_hor == '0'):
                action_prob[0, 3] = 1
            elif (current_x != next_x) or (current_z != next_z):
                action_prob[0, 0] = 1

        return action_prob

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass
