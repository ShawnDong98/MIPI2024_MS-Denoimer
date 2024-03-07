import torch
import torch.nn as nn


def apply_swa(model: nn.Module,
              checkpoint_list: list,
              weight_list: list,
              strict: bool = True,
              device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              ):
    """

    :param model:
    :param checkpoint_list: 要进行swa的模型路径列表
    :param weight_list: 每个模型对应的权重
    :param strict: 输入模型权重与checkpoint是否需要完全匹配
    :return:
    """

    checkpoint_tensor_list = [torch.load(f, map_location=device)['model'] for f in checkpoint_list]

    new_state_dict = {}

    for name, param in model.named_parameters():
        try:
            new_state_dict[name] = sum([ckpt[name] * w for ckpt, w in zip(checkpoint_tensor_list, weight_list)])
        except KeyError:
            if strict:
                raise KeyError(f"Can't match '{name}' from checkpoint")
            else:
                print(f"Can't match '{name}' from checkpoint")

    model.load_state_dict(new_state_dict)

    return model