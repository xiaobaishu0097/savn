import torch
import numpy as np

from utils.net_util import gpuify, toFloatTensor
from models.model_io import ModelInput

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
        if (self.episode.glove_reader[current_pos][CLASSES.index(self.episode.target_object)] != np.array([0, 0, 0, 0])).all():
            self.episode.current_det = True
        model_input.target_class_embedding = toFloatTensor(glove_embedding_tensor, self.gpu_id)
        model_input.action_probs = self.last_action_probs

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

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass
