from typing import List

import torch.nn as nn
from dlai_grader.grading import print_feedback, test_case
from torch.utils.data import Dataset
from unittests_utils import (
    TestBlockLayers,
    TestForwardPass,
    TestIRB,
    TestMakeBlock,
    TestTripleDataset,
)


def exercise_1(learner_class):
    def g():
        cases: List[test_case] = []

        class_name = "InvertedResidualBlock"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        t = test_case()
        if not issubclass(learner_class, nn.Module):
            t.failed = True
            t.msg = f"{class_name} must inherit from nn.Module"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        cases.append(t)

        # region = Check the initialization of the learner class =
        try:
            test_blocklayers = TestBlockLayers(learner_class)

            # check if it has all the modules
            t = test_case()
            got, want, failed = test_blocklayers.total_modules()
            if failed:
                t.failed = True
                t.msg = "The total number of modules is not correct"
                t.want = f"""The total number of modules should be {want}.
                Check that you have implemented the modules `expand` and `project`."""
                t.got = f"The total number of modules is {got}, which does not match."
                return [t]
            cases.append(t)

            # region == Check Expand Module ==

            # Existance
            t = test_case()
            got, want, failed = test_blocklayers.expand_presence()
            if failed:
                t.failed = True
                t.msg = "The expand module is not present"
                t.want = """The expand module not being `None`.
                Check that you have implemented the module `expand`."""
                t.got = "The expand module is `None`."
                return cases + [t]
            cases.append(t)

            # Total layers
            t = test_case()
            got, want, failed = test_blocklayers.expand_total_layers()
            if failed:
                t.failed = True
                t.msg = "The total number of layers in the expand module is not correct"
                t.want = (
                    f"The total number of layers in the expand module should be {want}."
                )
                t.got = f"The total number of layers in the expand module is {got}."
                return cases + [t]
            cases.append(t)

            # region === Checking sublayers ===
            # Convolution Layer
            t = test_case()
            got, want, failed = test_blocklayers.expand_first_layer_type()
            if failed:
                t.failed = True
                t.msg = (
                    "The first layer of the expand module is not of the correct type"
                )
                t.want = (
                    f"The first layer of the expand module should be a {want} layer"
                )
                t.got = f"The first layer of the expand module is {got}."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_blocklayers.expand_first_layer_output()
            if failed:
                t.failed = True
                t.msg = "The parameters of the Conv2d layer of the expand module are not correct."
                t.want = """Check that `in_channels`, `out_channels`, `kernel_size` and `bias` are set correctly."""
                t.got = "Some of the parameters are not set correctly."
            cases.append(t)

            # BatchNorm2d Layer
            t = test_case()
            got, want, failed = test_blocklayers.expand_second_layer_output()
            if failed:
                t.failed = True
                t.msg = "The parameters of the BatchNorm2d layer of the expand module are not correct."
                t.want = """Check that `num_features` is set correctly."""
                t.got = "Some of the parameters are not set correctly."
            cases.append(t)

            # ReLU Layer
            t = test_case()
            got, want, failed = test_blocklayers.expand_third_layer_type()
            if failed:
                t.failed = True
                t.msg = (
                    "The third layer of the expand module is not of the correct type"
                )
                t.want = (
                    f"The third layer of the expand module should be a {want} layer"
                )
                t.got = f"The third layer of the expand module is {got}."
            cases.append(t)

            # endregion ===

            # endregion ==

            # region == Check Project Module ==
            # Existance
            t = test_case()
            got, want, failed = test_blocklayers.project_presence()
            if failed:
                t.failed = True
                t.msg = "The project module is not present"
                t.want = """The project module not being `None`.
                Check that you have implemented the module `project`."""
                t.got = "The project module is `None`."
                return cases + [t]
            cases.append(t)

            # Total layers
            t = test_case()
            got, want, failed = test_blocklayers.project_total_layers()
            if failed:
                t.failed = True
                t.msg = (
                    "The total number of layers in the project module is not correct"
                )
                t.want = f"The total number of layers in the project module should be {want}."
                t.got = f"The total number of layers in the project module is {got}."
                return cases + [t]
            cases.append(t)

            # region === Checking sublayers ===
            # Convolution Layer
            t = test_case()
            got, want, failed = test_blocklayers.project_first_layer_type()
            if failed:
                t.failed = True
                t.msg = (
                    "The first layer of the project module is not of the correct type"
                )
                t.want = (
                    f"The first layer of the project module should be a {want} layer"
                )
                t.got = f"The first layer of the project module is {got}."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_blocklayers.project_first_layer_output()
            if failed:
                t.failed = True
                t.msg = "The parameters of the Conv2d layer of the project module are not correct."
                t.want = """Check that `in_channels`, `out_channels`, `kernel_size` and `bias` are set correctly."""
                t.got = "Some of the parameters are not set correctly."
            cases.append(t)

            # BatchNorm2d Layer
            t = test_case()
            got, want, failed = test_blocklayers.project_second_layer_output()
            if failed:
                t.failed = True
                t.msg = "The parameters of the BatchNorm2d layer of the project module are not correct."
                t.want = """Check that `num_features` is set correctly."""
                t.got = "Some of the parameters are not set correctly."
            cases.append(t)

            # endregion ===

            # endregion ==
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"The `InvertedResidualBlock` class is not defined correctly. Exception {e}"
            return cases + [t]
        # endregion =

        # region = Check Forward Pass =
        try:
            test_forward = TestForwardPass(learner_class)

            t = test_case()
            got, want, failed = test_forward.skip_definition()
            if failed:
                t.failed = True
                t.msg = "The skip connection variable is not defined correctly"
                t.want = """The skip connection variable should be defined.
                Check that you have fill `skip` with `x` in the forward method."""
                t.got = "`skip` is not defined correctly."
            cases.append(t)

            # expand
            t = test_case()
            got, want, failed = test_forward.expand_presence()
            if failed:
                t.failed = True
                t.msg = "The expand module is not being used within the forward method"
                t.want = """The first operation of the forward method should use the `.expand` module.
                Check that you have used the `.expand` module in the forward method."""
                t.got = "`.expand` is not used within the forward method."
            cases.append(t)

            # depthwise
            t = test_case()
            got, want, failed = test_forward.depthwise_presence()
            if failed:
                t.failed = True
                t.msg = (
                    "The depthwise module is not being used within the forward method"
                )
                t.want = """The second operation of the forward method should use the `.depthwise` module.
                Check that you have used the `.depthwise` module in the forward method."""
                t.got = "`.depthwise` is not used within the forward method."
            cases.append(t)

            # project
            t = test_case()
            got, want, failed = test_forward.project_presence()
            if failed:
                t.failed = True
                t.msg = "The project module is not being used within the forward method"
                t.want = """The third operation of the forward method should use the `.project` module.
                Check that you have used the `.project` module in the forward method."""
                t.got = "`.project` is not used within the forward method."
            cases.append(t)

            t = test_case()
            got, want, failed = test_forward.shortcut_use()
            if failed:
                t.failed = True
                t.msg = "The `.shortcut` module is present within the forward method"
                t.want = """If the `.shortcut` is not None, the `skip` variable should be modified using `.shortcut`.
                Check that the `skip` variable is modified using the `.shortcut` module in the forward method."""
                t.got = "When the `.shortcut` is not None, the `skip` variable is not modified correctly."
            cases.append(t)

            t = test_case()
            got_0, want_0, failed_0 = test_forward.skip_connection_use()
            got_1, want_1, failed_1 = test_forward.skip_connection_correct()
            failed = failed_0 or failed_1
            if failed:
                t.failed = True
                t.msg = "The `out` variable is not updated by adding `skip`"
                t.want = """`skip` should be added to `out` before returning it.
                Check that you have updated `out` by adding `skip` in the forward method."""
                t.got = "The `out` variable is not updated by adding `skip`"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"The forward method failed to execute. Error {e}"
            return cases + [t]

        # endregion=
        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_class):
    def g():
        cases: List[test_case] = []

        class_name = "MobileNetBackbone"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        t = test_case()
        if not issubclass(learner_class, nn.Module):
            t.failed = True
            t.msg = f"{class_name} must inherit from nn.Module"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        cases.append(t)

        try:
            # region = Check the _make_block method =
            test_make_block = TestMakeBlock(learner_class)

            t = test_case()
            got, want, failed = test_make_block.shortcut_presence()
            if failed:
                t.failed = True
                t.msg = "The `shortcut` variable is not defined correctly within `_make_block`"
                t.want = """If `in_channels` is different from `out_channels` or if `stride` is not 1,
                the `shortcut` variable should be defined as the `nn.Sequential` containing the Conv2d and BatchNorm2d layers."""
                t.got = "The `shortcut` variable is not used within the `_make_block` method."
            cases.append(t)
            # endregion =

            # region = Check the `blocks` module =
            test_irb = TestIRB(learner_class)

            t = test_case()
            got, want, failed = test_irb.total_IRB()
            if failed:
                t.failed = True
                t.msg = "The module `.blocks` is not being filled correctly"
                t.want = f"""The module `.blocks` should contain {want} `InvertedResidualBlock` modules, created using the `_make_block` method. Check that you have correctly made use of the `_make_block` method."""
                t.got = f"The total number of `InvertedResidualBlock` modules is {got}."
            cases.append(t)

            t = test_case()
            got, want, failed = test_irb.first_IRB_out_shape()
            if failed:
                t.failed = True
                t.msg = "The parameters passed in the first call of `_make_block` are not correct"
                t.want = f"""The first call of `_make_block` should have the following parameters:
                - `in_channels`: 16
                - `out_channels`: 24
                - `stride`: 2
                - `expansion_factor`: 3
                Check that you have correctly fill the inputs of the first call of `_make_block`."""
                t.got = f"The first `InvertedResidualBlock` of `.blocks` is not created correctly."
            cases.append(t)

            t = test_case()
            got, want, failed = test_irb.second_IRB_out_shape()
            if failed:
                t.failed = True
                t.msg = "The parameters passed in the second call of `_make_block` are not correct"
                t.want = f"""The second call of `_make_block` should have the following parameters:
                - `in_channels`: 24
                - `out_channels`: 32
                - `stride`: 2
                - `expansion_factor`: 3
                Check that you have correctly fill the inputs of the second call of `_make_block`."""
                t.got = f"The second `InvertedResidualBlock` of `.blocks` is not created correctly."
            cases.append(t)

            t = test_case()
            got, want, failed = test_irb.third_IRB_out_shape()
            if failed:
                t.failed = True
                t.msg = "The parameters passed in the third call of `_make_block` are not correct"
                t.want = f"""The third call of `_make_block` should have the following parameters:
                - `in_channels`: 32
                - `out_channels`: 64
                - `stride`: 2
                - `expansion_factor`: 6
                Check that you have correctly fill the inputs of the third call of `_make_block`."""
                t.got = f"The third `InvertedResidualBlock` of `.blocks` is not created correctly."
            cases.append(t)
            # endregion =

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = (
                "The `MobileNetBackbone` class is not defined correctly. Exception {e}"
            )
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_class):
    def g():
        cases: List[test_case] = []

        class_name = "TripleDataset"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        t = test_case()
        if not issubclass(learner_class, Dataset):
            t.failed = True
            t.msg = f"{class_name} must inherit from Dataset"
            t.want = Dataset
            t.got = learner_class.__base__
            return [t]
        cases.append(t)

        try:
            # region = Check `_get_positive_negative_indices` =
            test_td = TestTripleDataset(learner_class)

            t = test_case()
            got, want, failed = test_td.positive_indices()
            if failed:
                t.failed = True
                t.msg = "The `_get_positive_negative_indices` method does not work as expected"
                t.want = f"""`positive_index` should be the index of a sample being `anchor_label` its label.
                Check that the `positive_indices` are extracted from `.labels_to_indices` correctly."""
                t.got = "The `_get_positive_negative_indices` method does not work as expected"
                return [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_td.negative_indices()
            if failed:
                t.failed = True
                t.msg = "The `_get_positive_negative_indices` method does not work as expected"
                t.want = f"""`negative_index` should be the index of a sample with its label not being `anchor_label`.
                Check that the `negative_label` is completed correctly, being different from `anchor_label` and that
                `negative_indices` are extracted from `.labels_to_indices` correctly."""
                t.got = "The `_get_positive_negative_indices` method does not work as expected."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_td.positive_index_random()
            if failed:
                t.failed = True
                t.msg = "`random.choice` is not used to select the `positive_index`"
                t.want = """The `positive_index` should be selected using `random.choice` from the `positive_indices` list."""
                t.got = "`random.choice` is not used to select the `positive_index`."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_td.getitem_self_dataset()
            if failed:
                t.failed = True
                t.msg = "`__getitem__` is not working as expected"
                t.want = """`anchor_image, anchor_label` from the `idx` sample, should be extracted from `self.dataset`.
                Check that the `__getitem__` method is implemented correctly."""
                t.got = "`__getitem__` is not working as expected."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_td.getitem_positive_img()
            if failed:
                t.failed = True
                t.msg = "`__getitem__` is not working as expected"
                t.want = """`anchor_image` and `positive_image` should have the same label.
                Check that:
                - `positive_index, negative_index` are extracted using `_get_positive_negative_indices` correctly
                - `positive_image, _` are extracted from `self.dataset` using `positive_index` correctly."""
                t.got = (
                    "`anchor_image` and `positive_image` do not have the same label."
                )
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = test_td.getitem_negative_img()
            if failed:
                t.failed = True
                t.msg = "`__getitem__` is not working as expected"
                t.want = """`anchor_image` and `negative_image` should have the different label.
                Check that:
                - `positive_index, negative_index` are extracted using `_get_positive_negative_indices` correctly
                - `negative_image, _` are extracted from `self.dataset` using `negative_index` correctly."""
                t.got = "`anchor_image` and `negative_image` have the same label."
                return cases + [t]
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"The `TripleDataset` class is not defined correctly. Exception {e}"
            return cases + [t]

        return cases

        # endregion

    cases = g()
    print_feedback(cases)
