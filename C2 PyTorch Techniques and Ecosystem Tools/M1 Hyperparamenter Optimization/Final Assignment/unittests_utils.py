import json
import re
from collections import OrderedDict, defaultdict
from typing import List, Type, Union
from unittest.mock import Mock, patch

import optuna
import torch
import torch.nn as nn
from optuna.distributions import distribution_to_json, json_to_distribution


# region === Exercise 1 ===
def count_specific_layers(
    model: nn.Module, layer_types: Union[Type[nn.Module], List[Type[nn.Module]]]
) -> int:
    """Count specific types of layers in a model."""
    # Initialize layer count
    layer_count = 0

    # Iterate over all modules in the model
    for module in model.modules():
        # Check if the module is an instance of the specified layer type(s)
        if isinstance(module, layer_types):
            # Increment the count if it matches
            layer_count += 1

    # Return the total count
    # print(f"Total layers of type {layer_types}: {layer_count}")
    return layer_count


def get_mock_params(case: str) -> dict:
    if case == "1":
        mock_params_model = {
            "n_layers": 2,
            "n_filters": [16, 32],
            "kernel_sizes": [3, 3],
            "dropout_rate": 0.1,
            "fc_size": 64,
            "num_classes": 2,
        }

        mock_params_data_loader = {
            "resolution": 32,
            "batch_size": 8,
        }

    # expected_features_layers = {'Conv2d_0': torch.Size([1, 16, 32, 32]), 'BatchNorm2d_0': torch.Size([1, 16, 32, 32]), 'ReLU_0': torch.Size([1, 16, 32, 32]), 'MaxPool2d_0': torch.Size([1, 16, 16, 16]), 'Conv2d_1': torch.Size([1, 32, 16, 16]), 'BatchNorm2d_1': torch.Size([1, 32, 16, 16]), 'ReLU_1': torch.Size([1, 32, 16, 16]), 'MaxPool2d_1': torch.Size([1, 32, 8, 8])}
    expected_features_params = {
        "Conv2d": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": True,
        },
        "BatchNorm2d": {
            "num_features": 16,
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True,
        },
        "ReLU": {"inplace": False},
        "MaxPool2d": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        },
    }

    # expected_classifier_shapes =  OrderedDict([('L0_Dropout', ((1, 2048), 'Dropout')), ('L1_Linear', ((1, 64), 'Linear')), ('L2_ReLU', ((1, 64), 'ReLU')), ('L3_Dropout', ((1, 64), 'Dropout')), ('L4_Linear', ((1, 2), 'Linear'))])
    expected_classifier_params = [
        ("Dropout", {"p": 0.1, "inplace": False}),
        ("Linear", {"in_features": 2048, "out_features": 64, "bias": True}),
        ("ReLU", {"inplace": False}),
        ("Linear", {"in_features": 64, "out_features": 2, "bias": True}),
    ]

    # {'Conv2d_0': torch.Size([1, 16, 32, 32]), 'BatchNorm2d_0': torch.Size([1, 16, 32, 32]), 'ReLU_0': torch.Size([1, 16, 32, 32]), 'MaxPool2d_0': torch.Size([1, 16, 16, 16]), 'Conv2d_1': torch.Size([1, 32, 16, 16]), 'BatchNorm2d_1': torch.Size([1, 32, 16, 16]), 'ReLU_1': torch.Size([1, 32, 16, 16]), 'MaxPool2d_1': torch.Size([1, 32, 8, 8])}
    # OrderedDict([('dropout_c_1', ((1, 2048), 'Dropout')), ('f_c_1', ((1, 64), 'Linear')), ('relu_c', ((1, 64), 'ReLU')), ('dropout_c_2', ((1, 64), 'Dropout')), ('f_c_2', ((1, 2), 'Linear'))])

    total_layers = (
        mock_params_model["n_layers"] * 4
    )  # Conv2d + + BatchNorm2d + ReLU + MaxPool2d
    x_input = torch.randn(
        1,
        3,
        mock_params_data_loader["resolution"],
        mock_params_data_loader["resolution"],
    )

    # return mock_params_model, total_layers, expected_features_layers, x_input, expected_classifier_shapes
    return (
        mock_params_model,
        total_layers,
        expected_features_params,
        x_input,
        expected_classifier_params,
    )


def get_layer_output(name, layer_outputs):
    def hook(module, input, output):
        layer_outputs[name] = output.shape

    return hook


def add_hooks_features(model_features: nn.Module, layer_outputs: dict = None):
    for idx, block in enumerate(model_features.features):
        for layer in block:
            name = layer._get_name()
            layer.register_forward_hook(
                get_layer_output(name + f"_{idx}", layer_outputs)
            )


def get_sequence_features_shapes(model: nn.Module, x_input: torch.Tensor) -> dict:
    """
    Get the shapes of the features from each layer in the model.
    """
    layer_outputs = {}  # Reset the global dictionary to store outputs
    add_hooks_features(model, layer_outputs=layer_outputs)

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        _ = model(x_input)  # Example input shape

    return layer_outputs


def get_layer_shapes_and_types_sequential(
    sequential: nn.Sequential, input_tensor: torch.Tensor
):
    """
    Returns an OrderedDict mapping layer names to (output_shape, layer_type).
    If a layer has a name like '0', '1', etc. (default from Sequential), it will be replaced with
    '<LayerType>_<index>'. Otherwise, the provided name is used.
    """
    output_shapes = OrderedDict()

    def register_hook(name, layer):
        def hook(module, input, output):
            output_shapes[name] = (tuple(output.shape), type(module).__name__)

        layer.register_forward_hook(hook)

    for idx, (name, layer) in enumerate(sequential._modules.items()):
        if name.isdigit():
            layer_name = f"L{idx}_{type(layer).__name__}"
        else:
            layer_name = name
        register_hook(layer_name, layer)

    with torch.no_grad():
        sequential(input_tensor)

    return output_shapes

    # region == Extracting hyperparameters for each layer ==


def get_layer_hyperparams(layer):
    if isinstance(layer, torch.nn.Conv2d):
        return {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
            "bias": layer.bias is not None,
        }
    elif isinstance(layer, torch.nn.Linear):
        return {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "bias": layer.bias is not None,
        }
    elif isinstance(layer, torch.nn.BatchNorm2d):
        return {
            "num_features": layer.num_features,
            "eps": layer.eps,
            "momentum": layer.momentum,
            "affine": layer.affine,
            "track_running_stats": layer.track_running_stats,
        }
    elif isinstance(layer, torch.nn.MaxPool2d):
        return {
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "return_indices": layer.return_indices,
            "ceil_mode": layer.ceil_mode,
        }
    elif isinstance(layer, torch.nn.Dropout):
        return {"p": layer.p, "inplace": layer.inplace}
    elif isinstance(layer, torch.nn.ReLU):
        return {"inplace": layer.inplace}
    return {}


def extract_hyperparams_from_layers(
    sequential_model: nn.Sequential, use_idx=False
) -> defaultdict:
    """
    Extracts hyperparameters from each layer in a Sequential model.

    Args:
        sequential_model (nn.Sequential): The model to extract hyperparameters from.

    Returns:
        defaultdict: A dictionary where keys are layer names and values are dictionaries of hyperparameters.
    """
    hyperparams = defaultdict(dict)

    for idx, layer in enumerate(sequential_model):
        if use_idx:
            layer_name = f"{layer.__class__.__name__}_{idx}"
        else:
            layer_name = f"{layer.__class__.__name__}"
        hyperparams[layer_name] = get_layer_hyperparams(layer)

    return hyperparams


def extract_hyperparams_from_layers_bis(
    sequential_model: nn.Sequential, use_idx=False
) -> defaultdict:
    """
    Extracts hyperparameters from each layer in a Sequential model.

    Args:
        sequential_model (nn.Sequential): The model to extract hyperparameters from.

    Returns:
        defaultdict: A dictionary where keys are layer names and values are dictionaries of hyperparameters.
    """
    hyperparams = []

    for idx, layer in enumerate(sequential_model):
        if use_idx:
            layer_name = f"{layer.__class__.__name__}_{idx}"
        else:
            layer_name = f"{layer.__class__.__name__}"
        hyperparams.append((layer_name, get_layer_hyperparams(layer)))

    return hyperparams


def get_checks_classifier(learner_model, dummy_flattened_size=128):

    from unittest.mock import patch

    import torch.nn as nn

    with patch("torch.nn.Linear", wraps=nn.Linear) as mock_linear:
        learner_model._create_classifier(dummy_flattened_size)

    dict_info = {}
    dict_info["total_number_of_linear_layers"] = mock_linear.call_count

    for idx, call in enumerate(mock_linear.call_args_list):
        args, kwargs = call

        # Extract in_features and out_features from either args or kwargs
        in_features = kwargs.get("in_features") if "in_features" in kwargs else args[0]
        out_features = (
            kwargs.get("out_features") if "out_features" in kwargs else args[1]
        )

        dict_info[f"linear_layer_{idx+1}"] = {
            "in_features": in_features,
            "out_features": out_features,
        }

    learner_model.classifier = None
    return dict_info

    # endregion ==


# endregion ===

# region === Exercise 2 ===


def get_mock_fixed_trial(learner_search_space_func):
    # Create a fixed trial with predetermined values
    fixed_params = {
        "n_layers": 2,
        "n_filters_layer0": 16,
        "n_filters_layer1": 32,
        # "n_filters" : [16, 32],
        "kernel_size_layer0": 3,
        "kernel_size_layer1": 5,
        # "kernel_sizes": [3, 5],
        "dropout_rate": 0.001,
        "fc_size": 128,
        "learning_rate": 1e-3,
        "resolution": 32,
        "batch_size": 16,
    }

    trial = optuna.trial.FixedTrial(fixed_params)
    config = learner_search_space_func(trial)
    return config


def get_toy_trial(learner_search_space_func):
    def objective_toy(trial):
        params = learner_search_space_func(trial)
        return 0

    optuna.logging.disable_default_handler()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_toy, n_trials=1)
    trial = study.trials[0]
    return trial


def get_params_distributions(trial):

    keys = trial.params.keys()

    distributions_dict = {}

    for param in keys:
        # distributions_dict[param] = distribution_to_json(trial.distributions[param]) # uncomment to genereate expected distributions
        distributions_dict[param] = trial.distributions[param]
    return distributions_dict


def get_expected_params_distributions():
    expected_distributions_dict = {
        "n_layers": '{"name": "IntDistribution", "attributes": {"log": false, "step": 1, "low": 1, "high": 3}}',
        "n_filters_layer0": '{"name": "IntDistribution", "attributes": {"log": false, "step": 8, "low": 8, "high": 64}}',
        #'n_filters_layer1': '{"name": "IntDistribution", "attributes": {"log": false, "step": 8, "low": 8, "high": 64}}',
        "kernel_size_layer0": '{"name": "IntDistribution", "attributes": {"log": false, "step": 2, "low": 3, "high": 5}}',
        #'kernel_size_layer1': '{"name": "IntDistribution", "attributes": {"log": false, "step": 2, "low": 3, "high": 5}}',
        "dropout_rate": '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.1, "high": 0.5, "log": false}}',
        "fc_size": '{"name": "IntDistribution", "attributes": {"log": false, "step": 64, "low": 64, "high": 512}}',
        "learning_rate": '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.0001, "high": 0.01, "log": true}}',
        "resolution": '{"name": "CategoricalDistribution", "attributes": {"choices": [16, 32, 64]}}',
        "batch_size": '{"name": "CategoricalDistribution", "attributes": {"choices": [8, 16]}}',
    }

    # Convert each value from a JSON string to a Python dict
    # expected_distributions = {k: json_to_distribution(json.loads(v)) for k, v in expected_distributions_dict.items()}
    expected_distributions = {
        k: json_to_distribution(str(v)) for k, v in expected_distributions_dict.items()
    }

    return expected_distributions


# endregion ===


# region === Exercise 3 ===
def get_mock_params_ex3():
    mock_params_trial = optuna.trial.FixedTrial(
        {
            "n_layers": 2,
            "n_filters_layer0": 16,
            "n_filters_layer1": 32,
            "kernel_size_layer0": 3,
            "kernel_size_layer1": 3,
            "dropout_rate": 0.3,
            "fc_size": 128,
            "learning_rate": 0.001,
            "resolution": 32,
            "batch_size": 16,
        }
    )
    return mock_params_trial


def get_objects_during_trial(learner_obj_func, device, dataset_path):
    mock_trial = get_mock_params_ex3()

    # run the learner object function on fix params to get some objects created during the trial
    _ = learner_obj_func(
        trial=mock_trial,
        device=device,
        dataset_path=dataset_path,
        n_epochs=1,
        silent=True,
        test=True,
    )

    expected_parameters = mock_trial.params
    learner_objects = mock_trial.user_attrs

    transform = learner_objects["transform"]
    # train_loader = learner_objects['train_loader']
    model = learner_objects["model"]
    params_code = learner_objects["params_code"]
    return expected_parameters, transform, model, params_code


def remove_comments(code):
    # This regex pattern matches comments in the code
    pattern = r"#.*"

    # Use re.sub() to replace comments with an empty string
    code_without_comments = re.sub(pattern, "", code)

    # Split the code into lines, strip each line, and filter out empty lines
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Join the non-empty lines back into a single string
    return "\n".join(non_empty_lines)


# endregion ===

# region === Exercise 4 ===


def check_calls_model_parameters(learner_func):
    # Setup
    check_param1 = Mock()
    check_param1.requires_grad = True
    check_param1.numel.return_value = 100

    check_model = Mock()
    check_model.parameters.return_value = [check_param1]

    # Patch torch.numel to handle Mock objects
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # fallback

        mock_torch_numel.side_effect = side_effect

        # --- Run function ---
        result = learner_func(check_model)

        # Check that model.parameters() was called
        check = check_model.parameters.called
    return check


def check_iterates_parameters(learner_func):
    # --- Setup mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, Mock(requires_grad=False), p3]

    # Patch torch.numel to handle Mock objects
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # fallback

        mock_torch_numel.side_effect = side_effect

        # --- Run function ---
        learner_func(mock_model)

        # Check if torch.numel was called (since your solution uses torch.numel)
        # OR if param.numel() was called (for backward compatibility)
        check = mock_torch_numel.called or p1.numel.called or p3.numel.called

    return check


def check_requires_grad(learner_func):
    """
    Numerical check that the function correctly sums only the parameters that require gradients.
    """

    # --- Setup mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    # Mock for a non-trainable parameter
    p2 = Mock()
    p2.requires_grad = False
    p2.numel.return_value = 9999

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, p2, p3]

    expected_result = 100 + 300

    # Patch torch.numel to handle Mock objects
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 0  # fallback

        mock_torch_numel.side_effect = side_effect

        # --- Run function ---
        result = learner_func(mock_model)
        check = result == expected_result
    return check


def check_numel(learner_func):
    # --- Setup mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, Mock(requires_grad=False), p3]

    # Patch torch.numel to handle Mock objects
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # fallback

        mock_torch_numel.side_effect = side_effect

        # --- Run function ---
        learner_func(mock_model)

        # Check if torch.numel was called (for your solution)
        # OR if param.numel() was called (for backward compatibility)
        check = mock_torch_numel.called or (p1.numel.called and p3.numel.called)

    return check


# endregion ===
