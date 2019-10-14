from __future__ import division

import torch
from torch.autograd import Variable


def run_episode(player, args, total_reward, model_options, training):
    num_steps = args.num_steps

    for _ in range(num_steps):
        player.action(model_options, training)
        total_reward = total_reward + player.reward
        if player.done:
            break
    return total_reward


def new_episode(
        args,
        player,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        glove=None,
        img_file=None,
):
    player.episode.new_episode(args, scenes, possible_targets, targets, keep_obj, glove)
    player.reset_hidden()
    player.done = False


def a3c_loss(args, player, gpu_id, model_options):
    """ Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """
    R = torch.zeros(1, 1)
    if not player.done:
        _, output = player.eval_at_state(model_options)
        R = output.value.data

    det_frame = player.episode.det_frame
    # last_det = player.episode.last_det
    # current_det = player.episode.current_det
    # det_factor_value = 1
    det_factor_reward_base = 1
    # if (current_det == False) and (last_det == True):
    #     # det_factor_value = 0.5
    #     det_factor_reward = 2
    # elif (current_det == True) and (last_det == False):
    #     det_factor_reward = 0.5
    # det_factor_reward = torch.FloatTensor(det_factor_reward)
    det_factor_reward_base = float(det_factor_reward_base)
    # if gpu_id >= 0:
    #     with torch.cuda.device(gpu_id):
    #         # det_factor_value = det_factor_value.cuda()
    #         det_factor_reward = det_factor_reward.cuda()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    player.values.append(Variable(R))
    # player.optim_steps.append(player.episode.environment.controller.shortest_path_to_target(str(player.episode.environment.controller.state), player.episode.task_data[0])[1])
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()
    R = Variable(R)
    for i in reversed(range(len(player.rewards))):
        # if (det_frame != None) and (det_frame < i):
        #     det_factor_reward = det_factor_reward_base * (i - det_frame)
        # else:
        #     det_factor_reward = det_factor_reward_base
        det_factor_reward = det_factor_reward_base
        R = args.gamma * R + player.rewards[i] * det_factor_reward
        # R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        # advantage = R - (player.optim_steps[i] * -0.01 + 5)
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = (
                player.rewards[i] * det_factor_reward
                # player.rewards[i]
                + args.gamma * player.values[i + 1].data
                - player.values[i].data
        )

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = (
                policy_loss
                - player.log_probs[i] * Variable(gae)
                - args.beta * player.entropies[i]
        )

    return policy_loss, value_loss


def compute_learned_loss(args, player, gpu_id, model_options):
    loss_hx = torch.cat((player.hidden[0], player.last_action_probs), dim=1)
    learned_loss = {
        "learned_loss": player.model.learned_loss(
            loss_hx, player.learned_input, model_options.params
        )
    }
    player.learned_input = None
    # with torch.cuda.device(gpu_id):
    #     learned_loss['learned_loss'] = torch.tensor([0]).cuda()
    return learned_loss


def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
            player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()


def transfer_gradient_to_shared(gradient, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape)
            elif gpu_id < 0:
                param._grad = gradient[i]
            else:
                param._grad = gradient[i].cpu()

        i += 1


def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            # theta[name] = torch.tensor(
            #     param_copied,
            #     requires_grad=True,
            #     device=torch.device("cuda:{}".format(gpu_id)),
            # )
            # Changed for pythorch 0.4.1.
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta


def update_loss(sum_total_loss, total_loss):
    if sum_total_loss is None:
        return total_loss
    else:
        return sum_total_loss + total_loss


def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()


def SGD_step(theta, grad, lr):
    theta_i = {}
    j = 0
    for name, param in theta.items():
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param
        j += 1

    return theta_i


def get_scenes_to_use(player, scenes, args):
    if args.new_scene:
        return scenes
    return [player.episode.environment.scene_name]


def compute_loss(args, player, gpu_id, model_options):
    policy_loss, value_loss = a3c_loss(args, player, gpu_id, model_options)
    total_loss = policy_loss + 0.5 * value_loss
    return dict(total_loss=total_loss, policy_loss=policy_loss, value_loss=value_loss)


def end_episode(
        player, res_queue, title=None, episode_num=0, include_obj_success=False, **kwargs
):
    results = {
        "done_count": player.episode.done_count,
        "ep_length": player.eps_len,
        "success": int(player.success),
    }

    results.update(**kwargs)
    res_queue.put(results)


def get_bucketed_metrics(spl, best_path_length, success, done, arrive):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
    if done == 5:
        out["DONE success"] = success
    out["Arrive"] = arrive
    if success == True:
        out['Arrive AND Done'] = arrive
    return out


def compute_spl(player, start_state):
    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            # This is due to a rare known bug.
            continue

    if not player.success:
        return 0, best

    if best < float("inf"):
        return best / float(player.eps_len), best

    # This is due to a rare known bug.
    return 0, best


def generate_det_4_iou(det_bbox):
    # det_bbox = det_bbox.cpu()
    det_ious = torch.zeros((1, 4))
    image_areas = [
        [0, 0, 149, 149],
        [150, 0, 299, 149],
        [0, 149, 149, 299],
        [150, 149, 299, 299],
        # {'x1': 0, 'y1': 0, 'x2': 149, 'y2': 149},
        # {'x1': 150, 'y1': 0, 'x2': 299, 'y2': 149},
        # {'x1': 0, 'y1': 149, 'x2': 149, 'y2': 299},
        # {'x1': 150, 'y1': 149, 'x2': 299, 'y2': 299}
    ]

    for index, image_area in enumerate(image_areas):
        x_left = max(det_bbox[0], image_area[0])
        y_top = max(det_bbox[1], image_area[1])
        x_right = min(det_bbox[2], image_area[2])
        y_bottom = min(det_bbox[3], image_area[3])

        if (x_right < x_left) or (y_bottom < y_top):
            det_ious[0, index] = 0
            continue

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        area_1 = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        area_2 = (image_area[2] - image_area[0]) * (image_area[3] - image_area[1])

        det_ious[0, index] = intersection_area / float(area_1 + area_2 - intersection_area)

    return det_ious
