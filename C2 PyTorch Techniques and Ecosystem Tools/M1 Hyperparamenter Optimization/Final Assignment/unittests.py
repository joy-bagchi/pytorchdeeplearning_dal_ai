import inspect
from types import FunctionType

import torch
import torch.nn as nn
import unittests_utils
from dlai_grader.grading import print_feedback, test_case
from torchvision import transforms


def exercise_1(learner_class):

    def g():
        cases = []

        class_name = "FlexibleCNN"

        # region === general checks ===
        # Check if learner_class is a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} has incorrect type"
            t.want = f"a Python class called {class_name}"
            t.got = type(learner_class)
            return [t]

        # Check if learner_class inherits from nn.Module
        t = test_case()
        if learner_class.__base__ != nn.Module:
            t.failed = True
            t.msg = f"{class_name} didn't inherit from the correct class"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        # endregion ===

        # region === Check model features ===
        try:
            (
                mock_params_1,
                expected_total_layers_1,
                expected_features_params,
                x_input,
                expected_classifier_params,
            ) = unittests_utils.get_mock_params("1")
            expected_n_layers = mock_params_1["n_layers"]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            learner_model = learner_class(**mock_params_1).to(device)

            # region == Counting feature layers ==

            # region = Count Conv2d layers =
            learner_Conv2d_count = unittests_utils.count_specific_layers(
                learner_model, nn.Conv2d
            )

            t = test_case()
            if expected_n_layers != learner_Conv2d_count:
                t.failed = True
                t.msg = """The model should have as many Conv2d layers as the number of layers specified (n_layers).
                Please check that you are adding the corresponding Conv2d layer for each layer in the model"""
                t.want = "A total of n_layers Conv2d layers in the model"
                t.got = f"A total of {learner_Conv2d_count} Conv2d layers present in the model's architecture, what does not match with n_layers"
            cases.append(t)
            # endregion

            # region = Count BatchNorm2d layers =
            learner_BatchNorm_count = unittests_utils.count_specific_layers(
                learner_model, nn.BatchNorm2d
            )

            t = test_case()
            if expected_n_layers != learner_BatchNorm_count:
                t.failed = True
                t.msg = """The model should have as many BatchNorm2d layers as the number of layers specified (n_layers).
                Please check that you are adding the corresponding BatchNorm2d layer for each layer in the model"""
                t.want = "A total of n_layers BatchNorm2d layers in the model"
                t.got = f"A total of {learner_BatchNorm_count} BatchNorm2d layers present in the model's architecture, what does not match with n_layers"
            cases.append(t)
            # endregion

            # region = Count MaxPool2d layers =
            learner_MaxPool2d_count = unittests_utils.count_specific_layers(
                learner_model, nn.MaxPool2d
            )

            t = test_case()
            if expected_n_layers != learner_MaxPool2d_count:
                t.failed = True
                t.msg = """The model should have as many MaxPool2d layers as the number of layers specified (n_layers).
                Please check that you are adding the corresponding MaxPool2d layer for each layer in the model"""
                t.want = "A total of n_layers MaxPool2d layers in the model"
                t.got = f"A total of {learner_MaxPool2d_count} MaxPool2d layers present in the model's architecture, what does not match with n_layers"
            cases.append(t)
            # endregion

            # region = Count Total Amount of layers =
            learner_total_layers_count = (
                learner_Conv2d_count
                + learner_BatchNorm_count
                + learner_MaxPool2d_count
                + unittests_utils.count_specific_layers(learner_model, nn.ReLU)
            )

            t = test_case()
            if expected_total_layers_1 != learner_total_layers_count:
                t.failed = True
                t.msg = """The total amount of layers in the model does not match the expected total amount of layers, 4 x n_layers 
                (each of the n_layers convolutional_block should have 4 layers: Conv2d, BatchNorm2d, ReLU, MaxPool2d).
                Please check that you are adding the corresponding layers for each convolutional_block in the model"""
                t.want = "A total of 4 x n_layers layers in the model"
                t.got = f"{learner_total_layers_count} layers present in the model's architecture"
                cases.append(t)
                return cases
                # endregion

            # endregion

            # region == Check's parameters for feature layers ==
            learner_features_params_0 = unittests_utils.extract_hyperparams_from_layers(
                learner_model.features[0]
            )

            # region = Check parameters for the layers in the first convolutional_block =
            for layer, expected_params in expected_features_params.items():
                # Check if the parameters match
                for param, expected_value in expected_params.items():
                    t = test_case()
                    if param not in learner_features_params_0[layer]:
                        t.failed = True
                        t.msg = f"{param} parameter is not present in {layer} layer"
                        t.want = f"{param} parameter to be present in {layer} layer"
                        t.got = f"parameter {param} not found in {layer} layer"
                        cases.append(t)

                    t = test_case()
                    learner_param_value = learner_features_params_0[layer][param]
                    if learner_param_value != expected_value:
                        t.failed = True
                        if layer == "Conv2d" and param == "kernel_size":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that kernel_size=kernel_size"
                            t.want = f"parameter {param} is expected to be {param}"  # change this to match the msg
                            t.got = f"{learner_param_value[0]}"
                        elif layer == "Conv2d" and param == "padding":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that padding=padding"
                            t.want = f"parameter {param} = {param}"
                            t.got = f"{learner_param_value[0]}"
                        elif layer == "Conv2d" and param == "out_channels":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that out_channels=out_channels"
                            t.want = f"parameter {param} = {param}"
                            t.got = f"{learner_param_value}"
                        elif layer == "BatchNorm2d" and param == "num_features":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that num_features=out_channels"
                            t.want = f"parameter {param} = out_channels"
                            t.got = f"{learner_param_value}"
                        elif layer == "MaxPool2d" and param == "kernel_size":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that kernel_size=2"
                            t.want = f"parameter {param} = 2"
                            t.got = f"{learner_param_value}"
                        elif layer == "MaxPool2d" and param == "stride":
                            t.msg = f"{param} parameter value mismatch in {layer} layer. Please check that stride=2"
                            t.want = f"parameter {param} = 2"
                            t.got = f"{learner_param_value}"
                    cases.append(t)
            # endregion =

            # region = Check that in_channels=out_channels =
            t = test_case()
            out_channels_Conv2d_0 = learner_features_params_0["Conv2d"]["out_channels"]
            learner_features_params_1 = unittests_utils.extract_hyperparams_from_layers(
                learner_model.features[1]
            )
            in_channels_Conv2d_1 = learner_features_params_1["Conv2d"]["in_channels"]

            if out_channels_Conv2d_0 != in_channels_Conv2d_1:
                t.failed = True
                t.msg = f"in_channels of the second Conv2d layer is not equal to out_channels of the first Conv2d layer. Please check that you are setting in_channels=out_channels"
                t.want = f"in_channels of the second Conv2d layer to be equal to out_channels of the first Conv2d layer"
                t.got = f"in_channels of the second Conv2d layer: {in_channels_Conv2d_1}, out_channels of the first Conv2d layer: {out_channels_Conv2d_0}"
            cases.append(t)
        # endregion =
        # endregion ==
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"""Error during model execution: {e}.
            Check that the layers for each convolutional_block are created correctly."""
            t.want = """Model to process input without errors."""
            t.got = "Error"
            cases.append(t)
            return cases
        # endregion ===

        # region === Check classifier construction ===
        try:
            expected_flattened_size = 2048
            info_classifier = unittests_utils.get_checks_classifier(
                learner_model, expected_flattened_size
            )
            # print(learner_model.classifier)  # Uncomment for debugging

            # region == Check classifier linear layers ==
            # region = Check the amount of linear layers in the classifier ==
            t = test_case()
            total_linear_layers = info_classifier["total_number_of_linear_layers"]
            if total_linear_layers != 2:
                t.failed = True
                t.msg = f"Incorrect number of linear layers in the classifier. Please ensure that you have add 2 linear layers in the classifier."
                t.want = "2 linear layers in the classifier"
                t.got = f"{total_linear_layers} linear layers in the classifier"
                cases.append(t)
                return cases
            # endregion =

            # region = Check parameters of the linear layers in the classifier ==
            params_first_ll = info_classifier["linear_layer_1"]
            params_second_ll = info_classifier["linear_layer_2"]

            # Check in_features of the first linear layer
            t = test_case()
            if params_first_ll["in_features"] != expected_flattened_size:
                t.failed = True
                t.msg = (
                    "Incorrect in_features for the first linear layer in the classifier"
                )
                t.want = "For the first linear layer in the classifier, in_features = flattened_size."
                t.got = f"in_features of the first linear layer is {params_first_ll['in_features']}"
                cases.append(t)

            # Check out_features of the first linear layer
            t = test_case()
            if params_first_ll["out_features"] != learner_model.fc_size:
                t.failed = True
                t.msg = "Incorrect out_features for the first linear layer in the classifier"
                t.want = "For the first linear layer in the classifier, out_features = self.fc_size."
                t.got = f"out_features of the first linear layer is {params_first_ll['out_features']}"
                cases.append(t)

            # Check in_features of the second linear layer
            t = test_case()
            if params_second_ll["in_features"] != learner_model.fc_size:
                t.failed = True
                t.msg = "Incorrect in_features for the second linear layer in the classifier"
                t.want = "For the second linear layer in the classifier, in_features = self.fc_size."
                t.got = f"in_features of the second linear layer is {params_second_ll['in_features']}"
                cases.append(t)

            # Check out_features of the second linear layer
            t = test_case()
            if params_second_ll["out_features"] != learner_model.num_classes:
                t.failed = True
                t.msg = "Incorrect out_features for the second linear layer in the classifier"
                t.want = "For the second linear layer in the classifier, out_features = self.num_classes."
                t.got = f"out_features of the second linear layer is {params_second_ll['out_features']}"
                cases.append(t)

        # endregion =
        # endregion ==
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"""Error during classifier checks: {e}. \n
            Check that you are creating the classifier correctly in the _create_classifier() method. \n
            Check that self.classifier uses a Sequential container with: 
            nn.Dropout -> nn.Linear -> nn.ReLU -> nn.Dropout -> nn.Linear"""
            t.want = """Classifier to be constructed correctly."""
            t.got = "Error"
            cases.append(t)
            return cases
        # endregion ===

        # region === Forward pass checks ===
        try:
            x_input = x_input.to(device)
            learner_model_forward_output = learner_model(x_input)
            learner_flattened_size = learner_model._flattened_size
            expected_flattened_size = 2048

            # region == Check flattened size ==

            # region = flattened_size is not None ==
            t = test_case()
            if learner_model._flattened_size is None:
                t.failed = True
                t.msg = """ _flattened_size is None after the forward pass.
                Check that you have flattened x correctly (start_dim=1) and set ._flattened_size by extracting the size of x during the forward pass."""
                t.want = "_flattened_size to be set correctly after the forward pass."
                t.got = "flattened_size is None"
                cases.append(t)
                return cases
            # endregion =

            # region = flattened_size is correct ==
            # TODO: think an alternative way to provide a want message
            t = test_case()
            if learner_flattened_size != expected_flattened_size:
                t.failed = True
                t.msg = """_flattened_size is not correct. 
                Check that you have flattened x correctly (start_dim=1) and set ._flattened_size by extracting the size of x during the forward pass."""
                t.want = f"""_flattened_size to be {expected_flattened_size} after the forward pass."""
                t.got = f"_flattened_size is {learner_flattened_size}"
                cases.append(t)
                return cases
            # endregion =

            # endregion ==

            # region == Check that the classifier is not None ==
            t = test_case()
            if learner_model.classifier is None:
                t.failed = True
                t.msg = """.classifier is None after the forward pass.
                Check that you have completed the ._create_classifier() method during the forward pass."""
                t.want = ".classifier not to be None after the forward pass."
                t.got = ".classifier is None"
                cases.append(t)
                return cases
            # endregion ==

            # region == Check that the classifier is a Sequential object ==
            t = test_case()
            if not isinstance(learner_model.classifier, nn.Sequential):
                t.failed = True
                t.msg = """.classifier is not a Sequential object.
                Please check that you are completing the ._create_classifier() method correctly"""
                t.want = ".classifier to be a Sequential object."
                t.got = f".classifier is {type(learner_model.classifier)}"
                cases.append(t)
                return cases
        # endregion ==

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error during model forward pass: {e}. Check that you have flattened x correctly and set ._flattened_size = x.size(1)."
            t.want = "Model to process input without errors."
            t.got = "Error"
            cases.append(t)
        return cases

    # endregion ===

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []

        # region === general checks ===
        # check if learner_func is a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "design_search_space has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # endregion ===

        # region === check distributions of the optuna trial ===
        try:
            # extract the distributions defined within the learner_func, and the expected distributions
            expected_distributions = unittests_utils.get_expected_params_distributions()
            toy_trial = unittests_utils.get_toy_trial(learner_func)
            leaner_distributions = unittests_utils.get_params_distributions(toy_trial)

            # print(f"Expected distributions: {expected_distributions}")  # Uncomment for debugging
            # print(f"Leaner distributions: {leaner_distributions}")  # Uncomment for debugging

            # region == check if all the necessary parameters are present in the distributions ==
            missing_params = set(expected_distributions.keys()) - set(
                leaner_distributions.keys()
            )

            t = test_case()
            # if missing params is not empty, then the test should fail
            if missing_params:
                t.failed = True
                t.msg = """Some of the hyperparameters are missing in the search space.  \n Please check that you are completed the following hyperparameters: {', '.join(missing_params)}"""
                t.want = f"""At least the following hyperparameters to be present in the search space: {', '.join(expected_distributions.keys())}."""
                t.got = f"Missing hyperparameters: {', '.join(missing_params)}"
                cases.append(t)
                return cases
            # endregion ==

            # region == check if all the distributions are correct ==
            for param, expected_distribution in expected_distributions.items():
                t = test_case()
                if leaner_distributions[param] != expected_distribution:
                    t.failed = True
                    t.msg = f"Parameter '{param}' has incorrect distribution. \n Check that the trial.suggest_* method is the correct one (int or categorical or float) and that the parameters are set correctly."
                    t.want = f"""{expected_distribution}."""
                    t.got = f"{leaner_distributions[param]}"
                cases.append(t)
        # endregion ==
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error during model execution: {e}"
            t.want = "Model to process input without errors"
            t.got = "Error"
            cases.append(t)
        return cases

    # endregion ===

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []

        # region === general checks ===
        # == check if learner_func is a function ==
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "objective_function has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # endregion ===

        # region === check objects created during the objective function execution, within one trial ===
        # NOTE: we will check that the required objects were created and with the correct parameters
        try:
            learner_code = inspect.getsource(learner_func)
            cleaned_code = unittests_utils.remove_comments(learner_code)

            # region == check that design_search_space(trial) is called ==
            t = test_case()
            if "design_search_space(trial)" not in cleaned_code:
                t.failed = True
                t.msg = "design_search_space() is not called"
                t.want = "design_search_space(trial) is used to define params"
                t.got = "design_search_space(trial) is not called in objective_function"
                cases.append(t)
                return cases
            # endregion ==

            # region == check that the objects defined in the learner_func are created and uses the correct parameters ==
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            AIvsReal_path = "./AIvsReal_sampled"
            expected_parameters, transform, model, params_code = (
                unittests_utils.get_objects_during_trial(
                    learner_func, device=device, dataset_path=AIvsReal_path
                )
            )

            # region == A - check creation ==
            t = test_case()
            if not isinstance(transform, transforms.Compose):
                t.failed = True
                t.msg = "transform is not a transforms.Compose object. Please check that you are completing the `transform` variable with the transforms.Compose object."
                t.want = "transform to be a transforms.Compose object."
                t.got = f"type(transform) is {type(transform)}."
                cases.append(t)

            t = test_case()
            if not isinstance(model, nn.Module):
                t.failed = True
                t.msg = "model is not a nn.Module object. Please check that you are completing the `model` with a FlexibleCNN class that you have created in the exercise 1."
                t.want = "model to be a nn.Module object."
                t.got = f"type(model) is {type(model)}"
                cases.append(t)
                return cases
            # endregion ==

            # region == B - check parameters ==
            # region = i - the first transforms is Resize =
            learner_transform_0 = transform.transforms[0]

            t = test_case()
            if not isinstance(learner_transform_0, transforms.Resize):
                t.failed = True
                t.msg = "The first transform is not a Resize transform. Please check that you are completing the `transform` variable with the correct transforms."
                t.want = "The first transform to be a Resize transform."
                t.got = f"type of the first transform is {type(learner_transform_0)}"
                cases.append(t)
                return cases
            # endregion =

            # region = ii - parameters in transform =
            expected_resolution = expected_parameters["resolution"]
            learner_resolution = learner_transform_0.size[
                0
            ]  # Assuming the size is a tuple (height, width)
            t = test_case()
            if expected_resolution != learner_resolution:
                t.failed = True
                t.msg = "The resolution in the Resize transform does not match the expected resolution provided in params"
                t.want = "The resolution in the Resize transform to be equal to params['resolution']."
                t.got = (
                    f"The resolution in the Resize transform is {learner_resolution}."
                )
            cases.append(t)
            # endregion =

            # region = iii - some parameters in model =
            expected_n_layers = expected_parameters["n_layers"]
            expected_dropout_rate = expected_parameters["dropout_rate"]
            expected_fc_size = expected_parameters["fc_size"]

            learner_n_layers = len(model.features)
            learner_dropout_rate = model.dropout_rate
            learner_fc_size = model.fc_size

            t = test_case()
            if expected_n_layers != learner_n_layers:
                t.failed = True
                t.msg = "The number of layers in the model does not match the number of layers provided in params. Please check that you are completing FlexibleCNN with the parameters from params."
                t.want = "The number of layers in the model should match the number of layers provided in params."
                t.got = f"{learner_n_layers}"
            cases.append(t)

            t = test_case()
            if expected_dropout_rate != learner_dropout_rate:
                t.failed = True
                t.msg = "The dropout rate in the model does not match the dropout rate provided in params. Please check that you are completing FlexibleCNN with the parameters from params."
                t.want = "The dropout rate in the model should match the dropout rate provided in params."
                t.got = f"{learner_dropout_rate}"
            cases.append(t)

            t = test_case()
            if expected_fc_size != learner_fc_size:
                t.failed = True
                t.msg = "The fc_size in the model does not match the fc_size provided in params. Please check that you are completing FlexibleCNN with the parameters from params."
                t.want = "The fc_size in the model should match the fc_size provided in params."
                t.got = f"{learner_fc_size}"
            cases.append(t)
            # endregion =
        # endregion ==
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error during model execution: {e}. \n Check that you are completing the transform and the model with the correct parameters from params."
            t.want = "Model to process input without errors."
            t.got = "Error"
            cases.append(t)
        return cases

    # endregion ===

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []

        # region === 1 - general checks ===
        # check if learner_func is a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "get_trainable_params has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # endregion ===

        # region === 2 -check that certain objects are called within the source code of the learner_func ===

        try:
            check_1 = unittests_utils.check_calls_model_parameters(learner_func)
            check_2 = unittests_utils.check_iterates_parameters(learner_func)
            check_3 = unittests_utils.check_numel(learner_func)
            check_4 = unittests_utils.check_requires_grad(learner_func)

            t = test_case()
            if not check_1:
                t.failed = True
                t.msg = "model.parameters() is not called within `get_trainable_parameters` function"
                t.want = "model.parameters() to be called to obtain model parameters"
                t.got = "model.parameters() is not present"
                return cases + [t]

            t = test_case()
            if not check_2 or not check_3:
                t.failed = True
                t.msg = "The function does not iterate through model parameters or \n `total_trainable_parameters` is not computed correctly"
                t.want = "The function to iterate through model parameters and \n `total_trainable_parameters` to be updated using param.numel() or torch.numel(param)."
                t.got = "Wrong iteration through model parameters or wrong computation of total_trainable_parameters"
                return cases + [t]

            t = test_case()
            if not check_4:
                t.failed = True
                t.msg = "Check that `total_trainable_parameters` only includes parameters with `requires_grad=True`"
                t.want = "The function to check for param.requires_grad within the iteration through model parameters"
                t.got = "Bigger than expected number of trainable parameters found"
            cases.append(t)

        except Exception:
            t = test_case()
            t.failed = True
            t.msg = f"Error when executing `get_trainable_parameters`. \n Check that you are completing the model_parameters and total_trainable_parameters correctly."
            t.want = "To run the function without errors."
            t.got = "Error"
            cases.append(t)
        return cases

    # endregion ===
    cases = g()
    print_feedback(cases)
