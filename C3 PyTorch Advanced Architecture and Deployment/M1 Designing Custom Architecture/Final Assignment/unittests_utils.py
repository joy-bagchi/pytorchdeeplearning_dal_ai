import torch
import torch.nn as nn
from torch import fx
from torch.utils.data import Dataset
from torchinfo import summary
from torchinfo.layer_info import LayerInfo

# region = General Check structure =


class TestBattery:
    def __init__(self, learner_class):
        self.learner_class = learner_class

        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _create_reference_checks(self):
        checks_dict = {}
        if self.reference_checks is not None:
            for key in self.reference_checks.keys():
                check_fcn = getattr(self, f"{key}", None)
                # run the corresponding method
                got, want, failed = check_fcn()
                checks_dict[key] = got

        return checks_dict


# endregion

# region = Exercise 1 =


# region == Extraction info nn model ==
def get_summary(model, input_size, listed=True):
    col_names = ["input_size", "output_size", "num_params"]
    row_settings = ("depth", "var_names")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ti_summary = summary(
        model,
        input_size=input_size,
        col_names=col_names,
        row_settings=row_settings,
        device=device,
    )

    if listed:
        ti_summary = ti_summary.summary_list

    return ti_summary


class LIExtractor:
    def __init__(self, ti_summary):
        self.summary = ti_summary

    def _count_layers(self, layer_info: LayerInfo):
        depth = layer_info.depth
        return sum(1 for child in layer_info.children if child.depth == depth + 1)

    def get_module(self, layer_name: str):
        layer_info = next(
            (layer for layer in self.summary if layer.var_name == layer_name), None
        )
        if not layer_info:
            return None
        return layer_info

    def get_module_info(self, layer_info: LayerInfo):
        return {
            "name": layer_info.var_name,
            "class_type": layer_info.class_name,
            "input_shape": layer_info.input_size,
            "output_shape": layer_info.output_size,
        }

    def get_layer_info(self, module_name, idx_layer):
        module = self.get_module(module_name)
        if not module:
            return None

        layer = module.children[idx_layer] if idx_layer < len(module.children) else None

        layer_info = self.get_module_info(layer) if layer else None
        return layer_info


# endregion ==


# region == Test for the initialization of the model ==
def make_shortcut(in_channels, out_channels, stride):
    if in_channels != out_channels or stride != 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )
    return None


class TestBlockLayers(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):

        # Model
        shortcut = make_shortcut(3, 16, 1)

        config_model = {
            "in_channels": 3,
            "out_channels": 16,
            "stride": 1,
            "expansion_factor": 6,
            "shortcut": shortcut,
        }

        self.learner_model = self.learner_class(**config_model)

        self.input_size = (64, 3, 32, 32)

    def extract_info(self):
        # Get summary and initialize extractor
        ti_summary = get_summary(self.learner_model, self.input_size)
        self.extractor = LIExtractor(ti_summary)

    def get_reference_checks(self):

        self.reference_checks = {
            "total_modules": 4,
            "expand_presence": True,
            "expand_total_layers": 3,
            "expand_first_layer_type": "Conv2d",
            "expand_first_layer_output": [64, 18, 32, 32],
            "expand_second_layer_type": "BatchNorm2d",
            "expand_second_layer_output": [64, 18, 32, 32],
            "expand_third_layer_type": "ReLU",
            "project_presence": True,
            "project_total_layers": 2,
            "project_first_layer_type": "Conv2d",
            "project_first_layer_output": [64, 16, 32, 32],
            "project_second_layer_type": "BatchNorm2d",
            "project_second_layer_output": [64, 16, 32, 32],
        }

    def total_modules(self):
        check_name = "total_modules"

        # extract main model
        main_model = self.extractor.summary[0]

        # Count total modules
        got = self.extractor._count_layers(main_model)

        want = self.reference_checks[check_name]

        condition = got != want

        return got, want, condition

    # region == Expand module ==
    def expand_presence(self):
        check_name = "expand_presence"

        # Check if it has the expand module
        expand_module = self.extractor.get_module("expand")

        if expand_module is None:
            got = False
        else:
            got = True

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def expand_total_layers(self):
        check_name = "expand_total_layers"

        expand_module = self.extractor.get_module("expand")

        got = self.extractor._count_layers(expand_module)
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    # Check the Conv2d layer
    def expand_first_layer_type(self):
        check_name = "expand_first_layer_type"

        info_conv2d = self.extractor.get_layer_info("expand", 0)
        got = info_conv2d["class_type"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def expand_first_layer_output(self):
        check_name = "expand_first_layer_output"

        info_conv2d = self.extractor.get_layer_info("expand", 0)
        got = info_conv2d["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    # Check the BatchNorm2d Layer
    def expand_second_layer_type(self):
        check_name = "expand_second_layer_type"

        info_batchnorm = self.extractor.get_layer_info("expand", 1)
        got = info_batchnorm["class_type"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def expand_second_layer_output(self):
        check_name = "expand_second_layer_output"

        info_batchnorm = self.extractor.get_layer_info("expand", 1)
        got = info_batchnorm["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def expand_third_layer_type(self):
        check_name = "expand_third_layer_type"

        info_conv2d = self.extractor.get_layer_info("expand", 2)
        got = info_conv2d["class_type"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    # endregion ==

    # region == Project module ==
    def project_presence(self):
        check_name = "project_presence"

        # Check if it has the project module
        project_module = self.extractor.get_module("project")

        if project_module is None:
            got = False
        else:
            got = True

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_total_layers(self):
        check_name = "project_total_layers"

        project_module = self.extractor.get_module("project")

        got = self.extractor._count_layers(project_module)
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_first_layer_type(self):
        check_name = "project_first_layer_type"

        info_conv2d = self.extractor.get_layer_info("project", 0)
        got = info_conv2d["class_type"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_first_layer_output(self):
        check_name = "project_first_layer_output"

        info_conv2d = self.extractor.get_layer_info("project", 0)
        got = info_conv2d["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_second_layer_type(self):
        check_name = "project_second_layer_type"

        info_batchnorm = self.extractor.get_layer_info("project", 1)
        got = info_batchnorm["class_type"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_second_layer_output(self):
        check_name = "project_second_layer_output"

        info_batchnorm = self.extractor.get_layer_info("project", 1)
        got = info_batchnorm["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    # endregion ==


# endregion ==


# region == Extractor for the forward pass ==
def get_fx_forward_summary(model, input_tensor):
    fx_graph = fx.symbolic_trace(model)
    out = fx_graph(input_tensor)

    operations = []

    for node in fx_graph.graph.nodes:
        operations.append(node)

    return operations


class TestForwardPass(TestBattery):

    def __init__(self, learner_class, reference_available=True):
        super().__init__(learner_class)

    def _get_reference_inputs(self):
        # Model
        shortcut = make_shortcut(3, 16, 1)

        config_model = {
            "in_channels": 3,
            "out_channels": 16,
            "stride": 1,
            "expansion_factor": 6,
            "shortcut": shortcut,
        }

        self.learner_model = self.learner_class(**config_model)

    def extract_info(self):
        input_x = torch.randn(64, 3, 32, 32)

        self.fx_graph = get_fx_forward_summary(self.learner_model, input_x)

    def get_reference_checks(self):
        self.reference_checks = {
            "skip_definition": True,
            "expand_presence": True,
            "depthwise_presence": True,
            "project_presence": True,
            "shortcut_use": True,
            "skip_connection_use": True,
            "skip_connection_correct": True,
        }

    # === Checks ===

    def skip_definition(self):
        check_name = "skip_definition"

        for operation in self.fx_graph:
            if operation.name == "x":
                got = True
                break
            got = False

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def expand_presence(self):
        check_name = "expand_presence"
        got = False
        for operation in self.fx_graph:
            if operation.name == "expand_0":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def project_presence(self):
        check_name = "project_presence"
        got = False
        for operation in self.fx_graph:
            if operation.name == "project_0":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def depthwise_presence(self):
        check_name = "depthwise_presence"
        got = False
        for operation in self.fx_graph:
            if operation.name == "depthwise_0":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def shortcut_use(self):
        check_name = "shortcut_use"
        got = False
        for operation in self.fx_graph:
            if operation.name == "shortcut":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def skip_connection_use(self):
        check_name = "skip_connection_use"
        got = False
        for operation in self.fx_graph:
            if operation.name == "add":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def skip_connection_correct(self):
        check_name = "skip_connection_correct"
        got = False
        for operation in self.fx_graph:
            name = operation.name
            args = operation.args

            if name == "add":
                for op in args:
                    if hasattr(op, "name") and op.name == "shortcut":
                        got = True
                        break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


# endregion ==


# endregion =

# region = Exercise 2 =


class TestMakeBlock(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):

        # Model
        ex2_model = self.learner_class()

        # Block
        mb_dict = {
            "in_channels": 16,
            "out_channels": 24,
            "stride": 2,
            "expansion_factor": 3,
        }

        self.learner_model = ex2_model._make_block(**mb_dict)

        self.input_size = (32, 16, 32, 32)

    def extract_info(self):
        # Get summary and initialize extractor
        ti_summary = get_summary(self.learner_model, self.input_size)
        self.extractor = LIExtractor(ti_summary)

    def get_reference_checks(self):

        self.reference_checks = {
            "shortcut_presence": True,
            # "shortcut_total_layers": 2,
            # "shortcut_first_layer_type": "Conv2d",
            # "shortcut_first_layer_output_shape": [32, 24, 16, 16],
            # "shortcut_second_layer_type": "BatchNorm2d",
            # "shortcut_second_layer_output_shape": [32, 24, 16, 16],
        }

    def shortcut_presence(self):
        check_name = "shortcut_presence"

        got = False
        for layer in self.extractor.summary:
            if layer.var_name == "shortcut":
                got = True
                break
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


class TestIRB(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):

        # Model
        self.learner_model = self.learner_class().blocks

        self.input_size = (32, 16, 32, 32)

    def extract_info(self):
        # Get summary and initialize extractor
        ti_summary = get_summary(self.learner_model, self.input_size)
        self.extractor = LIExtractor(ti_summary)

    def get_reference_checks(self):

        self.reference_checks = {
            "total_IRB": 3,
            "first_IRB_out_shape": [32, 24, 16, 16],
            "second_IRB_out_shape": [32, 32, 8, 8],
            "third_IRB_out_shape": [32, 64, 4, 4],
        }

    def total_IRB(self):
        check_name = "total_IRB"

        self.IRBs = [
            layer
            for layer in self.extractor.summary
            if layer.class_name == "InvertedResidualBlock"
        ]
        got = len(self.IRBs)

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def first_IRB_out_shape(self):
        check_name = "first_IRB_out_shape"

        first_IRB = self.IRBs[0]
        got = self.extractor.get_module_info(first_IRB)["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def second_IRB_out_shape(self):
        check_name = "second_IRB_out_shape"

        second_IRB = self.IRBs[1]
        got = self.extractor.get_module_info(second_IRB)["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def third_IRB_out_shape(self):
        check_name = "third_IRB_out_shape"

        third_IRB = self.IRBs[2]
        got = self.extractor.get_module_info(third_IRB)["output_shape"]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


# endregion =

# region = Exercise 3 =


# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self):
        self.data = list(range(10))  # pretend "images" are just integers
        self.labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]  # some class distribution
        self.classes = list(set(self.labels))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class TestTripleDataset(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):
        self.dummy_dataset = DummyDataset()

    def extract_info(self):
        self.learner_dataset = self.learner_class(self.dummy_dataset)

        self.labels_to_indices = self.learner_dataset.labels_to_indices

    def get_reference_checks(self):
        self.reference_checks = {
            "positive_indices": True,
            "negative_indices": True,
            "positive_index_random": [True, True, True],
            "getitem_self_dataset": 7,
            "getitem_positive_img": True,
            "getitem_negative_img": True,
        }

    def positive_indices(self):
        check_name = "positive_indices"

        repetitions = 10
        checks = []

        for _ in range(repetitions):
            for anchor_label in self.labels_to_indices.keys():
                # use `._get_positive_negative_indices` for each anchor_label
                positive_index, negative_index = (
                    self.learner_dataset._get_positive_negative_indices(anchor_label)
                )

                # Check that the positive index is in the correct label
                correct = positive_index in self.labels_to_indices[anchor_label]
                checks.append(correct)

        got = all(checks)

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def negative_indices(self):
        check_name = "negative_indices"

        repetitions = 10
        checks = []

        for _ in range(repetitions):
            for anchor_label in self.labels_to_indices.keys():
                # use `._get_positive_negative_indices` for each anchor_label
                positive_index, negative_index = (
                    self.learner_dataset._get_positive_negative_indices(anchor_label)
                )

                # check that the negative index is not in the positive indices
                correct = negative_index not in self.labels_to_indices[anchor_label]
                checks.append(correct)

        got = all(checks)

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def positive_index_random(self):
        check_name = "positive_index_random"

        repetitions = 100
        checks = []

        for anchor_label in self.labels_to_indices.keys():
            experiments = []
            for _ in range(repetitions):
                positive_index, negative_index = (
                    self.learner_dataset._get_positive_negative_indices(anchor_label)
                )
                experiments.append(positive_index)

            # check if the positive indexes recollected cover all the possible indexes for that anchor label
            all_present = set(experiments) == set(self.labels_to_indices[anchor_label])
            checks.append(all_present)
        got = checks

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def getitem_self_dataset(self):
        check_name = "getitem_self_dataset"

        idx = 7

        got = self.learner_dataset[idx][0]

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def getitem_positive_img(self):
        check_name = "getitem_positive_img"

        idx = 7

        learner_triple = self.learner_dataset[idx]

        anchor_img = learner_triple[0]
        anchor_label = self.dummy_dataset.labels[anchor_img]

        positive_img = learner_triple[1]
        positive_label = self.dummy_dataset.labels[positive_img]

        same_label = anchor_label == positive_label

        got = same_label

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition

    def getitem_negative_img(self):
        check_name = "getitem_negative_img"

        idx = 7

        learner_triple = self.learner_dataset[idx]

        anchor_img = learner_triple[0]
        anchor_label = self.dummy_dataset.labels[anchor_img]

        negative_img = learner_triple[2]
        negative_label = self.dummy_dataset.labels[negative_img]

        different_label = anchor_label != negative_label
        got = different_label

        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


# endregion =
