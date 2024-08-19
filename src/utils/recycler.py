import logging
import math
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def _kaiming_uniform_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the weights of a layer according to the Kaiming uniform scheme."""

    nn.init.kaiming_uniform_(layer.weight.data[mask, ...], a=math.sqrt(5))


@torch.no_grad()
def _orthogonal_reinit(layer: nn.Module, mask: torch.Tensor) -> None:
    """Partially re-initializes the weights of a layer according to the orthogonal scheme."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data[mask, ...])
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(layer.weight.data[mask, ...], gain)


@torch.no_grad()
def _lecun_normal_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the weights of a layer according to the Lecun normal scheme."""

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    torch.nn.init.trunc_normal_(layer.weight[mask])
    layer.weight[mask] *= stddev


@torch.inference_mode()
def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Get the activations of a layer with relu nonlinearity.
        ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
        """
        activations[name] = F.relu(output)

    return hook


@torch.inference_mode()
def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> List[torch.Tensor]:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []

    # Last activation are the q-values, which are never reset
    for name, activation in list(activations.items())[:-1]:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
    return masks


def _reset_dormant_neurons(layers, redo_masks: List[torch.Tensor], use_lecun_init: bool):
    """Re-initializes the dormant neurons of a model."""
    reinit_function = _lecun_normal_reinit if use_lecun_init else _orthogonal_reinit

    ingoing_layers = layers[:-1]
    outgoing_layers = layers[1:]
    logger.debug(f"redo masks shapes {[i.shape for i in redo_masks]}")

    # Sanity checks
    # assert "q" not in [name for name, _ in ingoing_layers], "The q layer should not be reset."
    assert "conv1" not in [name for name, _ in outgoing_layers], "The first conv layer should never be set to 0."
    assert (
        len(ingoing_layers) == len(outgoing_layers) == len(redo_masks)
    ), f"The number of layers and masks should match the number of masks. Got {len(ingoing_layers)}, {len(outgoing_layers)}, {len(redo_masks)}."

    # Reset the ingoing weights
    # Here the mask size always matches the layer weight size
    for (_, layer), mask in zip(ingoing_layers, redo_masks, strict=True):
        if not torch.all(~mask):
            # The initialization scheme is the same for conv2d and linear
            # 1. Reset the ingoing weights using the initialization distribution
            reinit_function(layer, mask)

    # Set the outgoing weights to 0
    i = 0
    for (name, layer), (next_name, next_layer), mask in zip(ingoing_layers, outgoing_layers, redo_masks, strict=True):
        logger.debug(f"layer {name} next_layer {next_name} mask {mask.shape}")
        if torch.all(~mask):
            # No dormant neurons in this layer
            i += 1
            continue
        elif isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Special case: Transition from conv to linear layer
            # Reset the outgoing weights to 0 with a mask created from the conv filters
            num_repetition = next_layer.weight.data.shape[0] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repetition)
            next_layer.weight.data[linear_mask, :].data.fill_(0)
        else:
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            # in critic first mask may be too short as we concatenate with action.
            mask_shape = mask.shape[0]
            next_layer_shape = next_layer.weight.data.shape[1]
            if i == 0 and mask_shape < next_layer_shape:
                # mask is too short
                mask = torch.cat(
                    [mask, torch.zeros(next_layer_shape - mask_shape, dtype=torch.bool, device=mask.device)]
                )

            next_layer.weight.data[:, mask, ...].data.fill_(0)

        i += 1

    # return model


def _reset_adam_moments(
    optimizer: optim.Adam,
    reset_masks: List[torch.Tensor],
    model_types_names: List[List[str]],
    offset: Optional[int] = None,
) -> None:
    """
    Resets the moments of an Adam optimizer for the weights and biases of a model.
    :param optimizer: The optimizer to reset the moments for.
    :param reset_masks: The masks to reset the moments for.
    :param model_types_names: The names of the layers in the model. Norm layers are grouped together within
    a list with the previous linear or conv layer.
    :param offset: The offsets for the layers. This is used to skip layers in the model. Used when resetting critic
    which consists of an ensemble of critics.
    """

    assert isinstance(optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"
    assert len(reset_masks) == len(model_types_names), "The number of masks should match the number of layers."

    if not optimizer.state_dict()["state"].keys():
        logger.warning("No optimizer state dict found. Skipping moment reset.")
        # return optimizer

    # print optimizer shapes
    optimizer_state_keys = optimizer.state_dict()["state"].keys()
    logger.debug(f"Offset {offset}")
    logger.debug(f"Model types names {model_types_names}")
    logger.debug(f"optimizer state keys {optimizer_state_keys}")
    for key in optimizer_state_keys:
        logger.debug(f"exp_avg shape {optimizer.state_dict()['state'][key]['exp_avg'].shape}")
        logger.debug(f"exp_avg_sq shape {optimizer.state_dict()['state'][key]['exp_avg_sq'].shape}")

    # print masks shapes
    logger.debug(f"reset masks shapes {[i.shape for i in reset_masks]}")

    i = 0
    added_offset = False
    for mask, layer_names in zip(reset_masks, model_types_names, strict=True):
        # Reset the moments for the weights
        # NOTE: I don't think it's possible to just reset the step for moment that's being reset
        # NOTE: As far as I understand the code, they also don't reset the step count
        for name in layer_names:
            optimizer.state_dict()["state"][i]["exp_avg"][mask, ...] = 0.0
            optimizer.state_dict()["state"][i]["exp_avg_sq"][mask, ...] = 0.0

            # Reset the moments for the bias
            # optimizer.state_dict()["state"][i*2 + 1]['step'] = torch.tensor(0.0)
            # TODO should be reset here for bias?
            # optimizer.state_dict()["state"][i + 1]["exp_avg"][mask] = 0.0
            # optimizer.state_dict()["state"][i + 1]["exp_avg_sq"][mask] = 0.0
            i += 2

        # Skip to critic
        if (offset is not None) and (not added_offset):
            i += offset
            added_offset = True

    # return optimizer


def _get_modules_names_in_optimizer(named_modules: List[Tuple[str, nn.Module]]) -> List[List[str]]:
    """
    Returns the names of all modules in the optimizer. Filters the named_modules() output to only include
    linear layers, conv layers and norm layers. More layer types can be added if necessary.
    Norm layers are grouped with the previous layer type.

    :param model: The model to extract the module names from.
    :return: A list of lists of module types names.
    """
    module_types_names: List[List[str]] = []
    for name, module in named_modules:
        if isinstance(module, nn.Linear):
            module_types_names.append(["linear"])
        elif isinstance(module, nn.Conv2d):
            module_types_names.append(["conv"])
        # normalization layers
        elif isinstance(module, nn.LayerNorm):
            assert len(module_types_names) > 0, "Normalization layers should not be the first layer in the model."
            module_types_names[-1].append("layer_norm")
        elif isinstance(module, nn.BatchNorm2d):
            assert len(module_types_names) > 0, "Normalization layers should not be the first layer in the model."
            module_types_names[-1].append("batch_norm")

    return module_types_names


def reinitialize_dormant_neurons(
    model: nn.Module,
    optimizer: optim.Adam,
    masks: List[torch.Tensor],
    use_lecun_init: bool = False,
    named_modules=None,
    offset: Optional[int] = None,
) -> None:
    if named_modules is None:
        layers = [
            (name, i) for name, i in model.named_modules() if isinstance(i, nn.Linear) or isinstance(i, nn.Conv2d)
        ]
        model_types_names = _get_modules_names_in_optimizer(list(model.named_modules()))

    else:
        layers = [(name, i) for name, i in named_modules if isinstance(i, nn.Linear) or isinstance(i, nn.Conv2d)]
        model_types_names = _get_modules_names_in_optimizer(named_modules)

    _reset_dormant_neurons(layers, masks, use_lecun_init)
    # model = _reset_dormant_neurons(model, masks, use_lecun_init)
    # we do not analyze last layer.
    # optimizer = _reset_adam_moments(optimizer, masks, model_types_names[:-1])
    _reset_adam_moments(optimizer, masks, model_types_names[:-1], offset=offset)
    # return model, optimizer


def calculate_dormant_neurons(
    observations: Tuple[torch.Tensor, ...],
    model: nn.Module,
    tau: float,
    model_modules: Optional[List[Tuple[str, Any]]] = None,
    modules_names: Optional[List[str]] = None,
) -> dict:
    """
    Checks the number of dormant neurons for a given model.

    Returns the number of dormant neurons.
    """
    with torch.inference_mode():
        activations = {}
        activation_getter = partial(_get_activation, activations=activations)

        # Register hooks for all Conv2d and Linear layers to calculate activations
        handles = []
        if model_modules is not None:
            assert modules_names is not None, "modules_names should be provided if model_modules is provided"
            for mod, name in zip(model_modules, modules_names, strict=True):
                if isinstance(mod[1], torch.nn.Conv2d) or isinstance(mod[1], torch.nn.Linear):
                    logger.debug(f"module {name}")
                    handles.append(mod[1].register_forward_hook(activation_getter(name)))
        else:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    logger.debug(f"module {name}")
                    handles.append(module.register_forward_hook(activation_getter(name)))

        # Calculate activations. Observations is a tuple o tensors depending whether actor or critic is considered.
        model_outputs = model(*observations)
        logger.debug(f"activations keys {activations.keys()}")

        # Remove the hooks again
        for handle in handles:
            handle.remove()

        # Masks for tau=0 logging
        zero_masks = _get_redo_masks(activations, 0.0)
        zero_count = sum([torch.sum(mask) for mask in zero_masks])
        zero_fraction = (zero_count / sum([torch.numel(mask) for mask in zero_masks])) * 100

        # Calculate the masks actually used for resetting
        logger.debug(activations.keys())
        masks = _get_redo_masks(activations, tau)
        dormant_count = sum([torch.sum(mask) for mask in masks])
        dormant_fraction = (dormant_count / sum([torch.numel(mask) for mask in masks])) * 100

        logger.debug(f"len redo masks {len(masks)}")
        logger.debug(f"redo shapes {[i.shape for i in masks]}")

        return {
            "zero_fraction": zero_fraction,
            "zero_count": zero_count,
            "dormant_fraction": dormant_fraction,
            "dormant_count": dormant_count,
            "model_outputs": model_outputs,
            "masks": masks,
        }
