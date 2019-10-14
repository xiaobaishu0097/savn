class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(
        self, state=None, hidden=None, target_class_embedding=None, action_probs=None, det_relation=None, optim_steps=None, det_his=None, det_cur=None
    ):
        self.state = state
        self.hidden = hidden
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs
        self.det_relation = det_relation
        self.optim_steps = optim_steps
        self.det_his = det_his
        self.det_cur = det_cur


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None):

        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
