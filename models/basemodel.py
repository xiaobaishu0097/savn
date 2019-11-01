from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        # target_embedding_sz = args.glove_dim
        target_embedding_sz = 95
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        # self.embed_glove = nn.Linear(19*262, 64*7*7)
        # self.embed_glove = nn.Linear(target_embedding_sz, 64)
        # self.embed_glove = nn.Conv2d(256, 64, 1, 1)
        self.embed_action = nn.Linear(action_space, 10)
        # self.embed_action = nn.Linear(10, 10)

        self.detection_appearance_linear_1 = nn.Linear(512, 128)
        self.detection_other_info_linear_1 = nn.Linear(6, 19)
        self.detection_other_info_linear_2 = nn.Linear(19, 19)
        self.detection_appearance_linear_2 = nn.Linear(128, 49)
        # self.graph = nn.Linear(19, 19)

        # pointwise_in_channels = 138
        pointwise_in_channels = 93

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        # self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        # self.critic_linear_2 = nn.Linear(72, 1)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        # self.critic_linear.weight.data = norm_col_init(
        #     self.critic_linear.weight.data, 1.0
        # )
        # self.critic_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_embedding_input, params):

        # action_embedding_input = action_probs
        # target = target.view([95])
        # target = target.view([4978])
        target_appear = target[:, :512]
        target_info = target[:, 512:]

        if params is None:
            # glove_embedding = F.relu(self.embed_glove(target))
            # # glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)
            # glove_reshaped = glove_embedding.reshape(1, 64, 7, 7)

            target_appear = F.relu(self.detection_appearance_linear_1(target_appear))
            # target_info = F.relu(self.detection_other_info_linear(target_info))
            # target_embedding = torch.cat((target_appear, target_info), dim=1)
            target_info = F.relu(self.detection_other_info_linear_1(target_info))
            target_info = target_info.t()
            target_info = F.relu(self.detection_other_info_linear_2(target_info))
            target_info = target_info.t()
            target_appear = target_appear.t()
            target_appear = target_appear.mm(target_info)
            target_appear = target_appear.t()
            target_appear = F.relu(self.detection_appearance_linear_2(target_appear))
            target_embedding = target_appear.reshape(1, 19, 7, 7)
            # Graph structure
            # target_embedding = target_embedding.t()
            # target_embedding = F.relu(self.graph(target_embedding))
            # target_embedding = target_embedding.t()

            # target_embedding = target_embedding.reshape(1, 19, 7, 7)

            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(self.conv1(state))
            x = self.dropout(image_embedding)
            x = torch.cat((x, target_embedding, action_reshaped), dim=1)
            x = F.relu(self.pointwise(x))
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:
            target_appear = F.relu(
                F.linear(
                    target_appear,
                    weight=params["detection_appearance_linear_1.weight"],
                    bias=params["detection_appearance_linear_1.bias"],
                )
            )
            # target_info = F.relu(
            #     F.linear(
            #         target_info,
            #         weight=params["detection_other_info_linear.weight"],
            #         bias=params["detection_other_info_linear.bias"],
            #     )
            # )
            # target_embedding = torch.cat((target_appear, target_info), dim=1)

            target_info = F.relu(
                F.linear(
                    target_info,
                    weight=params["detection_other_info_linear_1.weight"],
                    bias=params["detection_other_info_linear_1.bias"],
                )
            )
            target_info = target_info.t()
            target_info = F.relu(
                F.linear(
                    target_info,
                    weight=params["detection_other_info_linear_2.weight"],
                    bias=params["detection_other_info_linear_2.bias"],
                )
            )
            target_info = target_info.t()
            target_appear = target_appear.t()
            target_appear = target_appear.mm(target_info)
            target_appear = target_appear.t()
            target_appear = F.relu(
                F.linear(
                    target_appear,
                    weight=params["detection_appearance_linear_2.weight"],
                    bias=params["detection_appearance_linear_2.bias"],
                )
            )
            target_embedding = target_appear.reshape(1, 19, 7, 7)

            # Graph structure
            # target_embedding = target_embedding.t()
            # target_embedding = F.relu(
            #     F.linear(
            #         target_embedding,
            #         weight=params["graph.weight"],
            #         bias=params["graph.bias"],
            #     )
            # )
            # target_embedding = target_embedding.t()

            # target_embedding = target_embedding.reshape(1, 19, 7, 7)

            # glove_embedding = F.relu(
            #     F.linear(
            #         target,
            #         weight=params["embed_glove.weight"],
            #         bias=params["embed_glove.bias"],
            #     )
            # )

            # glove_embedding = F.avg_pool2d(
            #     F.conv2d(
            #         target,
            #         weight=params['embed_glove.weight'],
            #         bias=params['embed_glove.bias']
            #     ),
            #     (7, 7),
            #     # (1, 1)
            #     (3, 3),
            # )
            #
            # glove_embedding = F.conv2d(
            #         target,
            #         weight=params['embed_glove.weight'],
            #         bias=params['embed_glove.bias']
            # )

            # glove_embedding = torch.randn(1,64,7,7).cuda()

            # glove_embedding = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)
            # glove_embedding = glove_embedding.reshape(1, 64, 7, 7)

            action_embedding = F.relu(
                F.linear(
                    action_embedding_input,
                    weight=params["embed_action.weight"],
                    bias=params["embed_action.bias"],
                )
            )
            action_embedding = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(
                F.conv2d(
                    state,
                    weight=params["conv1.weight"],
                    bias=params["conv1.bias"]
                )
            )
            x = self.dropout(image_embedding)
            # x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = torch.cat((x, target_embedding, action_embedding), dim=1)

            x = F.relu(
                F.conv2d(
                    x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
                )
            )
            x = self.dropout(x)
            x = x.view(x.size(0), -1)

        return x, image_embedding

    def a3clstm(self, embedding, prev_hidden, params, det_his=None):
        if embedding.shape == (1, 64, 7, 7):
            embedding = embedding.view(embedding.size(0), -1)
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            # det_x = torch.cat((x, optim_steps), dim=1)
            # critic_out = self.critic_linear(x)
            x = self.critic_linear_1(x)
            # x = torch.cat((x, torch.unsqueeze(det_his, dim=0)), dim=1)
            critic_out = self.critic_linear_2(x)

        else:
            hx, cx = self._backend.LSTMCell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            x = hx

            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

            # det_x = torch.cat((x, optim_steps), dim=1)

            # critic_out = F.linear(
            #     x,
            #     weight=params["critic_linear.weight"],
            #     bias=params["critic_linear.bias"],
            # )

            x = F.linear(
                x,
                weight=params["critic_linear_1.weight"],
                bias=params["critic_linear_1.bias"],
            )
            # x = torch.cat((x, det_his), dim=1)
            critic_out = F.linear(
                x,
                weight=params["critic_linear_2.weight"],
                bias=params["critic_linear_2.bias"],
            )

            # critic_out = F.linear(
            #     det_x,
            #     weight=params["critic_linear.weight"],
            #     bias=params["critic_linear.bias"],
            # )


        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        params = model_options.params
        # det_his = torch.cat((model_input.det_his, model_input.det_cur))
        # det_relation = model_input.det_relation
        # optim_steps = model_input.optim_steps

        x, image_embedding = self.embedding(state, target, action_probs, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)
        # actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params, det_his)
        # actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params, det_relation)
        # actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params, optim_steps)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
