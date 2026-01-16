import inspect
from types import FunctionType

import torch
from torchvision.models import resnet50
import torch.nn as nn
from dlai_grader.grading import print_feedback, test_case
from torchvision import transforms

import helper_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exercise_1(learner_func):
    def g():
        cases = []

        # region === general checks ===
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "cnn_feature_hierarchy_demo has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # endregion

        try:
            import tempfile
            from PIL import Image
            import numpy as np
            import torch

            model = resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model = model.to(device).eval()

            # Create a dummy image to feed into the function
            img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
            
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                img.save(tmp.name)

                img = helper_utils.preprocess_image(tmp.name, device=device)

                # Call learner function
                result = learner_func(img, model)

            # Check output type
            t = test_case()
            if not isinstance(result, dict):
                t.failed = True
                t.msg = "Function should return a dictionary."
                t.want = "Dictionary of activations (str → torch.Tensor)"
                t.got = type(result)
                cases.append(t)
                return cases

            # Check expected keys in output
            expected_keys = {"conv1", "layer1", "layer2", "layer3", "layer4"}
            returned_keys = set(result.keys())
            missing = expected_keys - returned_keys

            t = test_case()
            if missing:
                t.failed = True
                t.msg = f"Missing expected keys in returned dictionary: {missing}"
                t.want = f"Keys: {expected_keys}"
                t.got = f"Returned keys: {returned_keys}"
            cases.append(t)

            # Check tensor types and dimensions
            for k, v in result.items():
                t = test_case()
                if not isinstance(v, torch.Tensor):
                    t.failed = True
                    t.msg = f"Value for key '{k}' is not a torch.Tensor."
                    t.want = torch.Tensor
                    t.got = type(v)
                elif v.dim() != 4:
                    t.failed = True
                    t.msg = f"Tensor for key '{k}' must be 4D (B, C, H, W)."
                    t.want = "4D tensor"
                    t.got = f"{v.shape}"
                cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing function: {e}"
            t.want = "Execution without errors"
            t.got = "Exception raised"
            cases.append(t)
        # endregion

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []

        # region === general checks ===
        from types import FunctionType
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "feature_map_strip has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        # endregion

        # region === functional test with toy image ===
        try:
            import tempfile
            import numpy as np
            from PIL import Image
            import torch
            from torchvision.models import resnet50

            # build a dummy ResNet-50 and set to eval
            model = resnet50(weights=None).eval()

            # create a random RGB image
            img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))

            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                img.save(tmp.name)
                img = helper_utils.preprocess_image(tmp.name)
                # call learner function
                result = learner_func(img, model)

            # check return type is a list
            t = test_case()
            if not isinstance(result, list):
                t.failed = True
                t.msg = "function should return a list of upsampled tensors"
                t.want = "list"
                t.got = type(result)
                cases.append(t)
                return cases
            cases.append(t)

            # check list length
            t = test_case()
            if len(result) != 5:
                t.failed = True
                t.msg = f"expected 5 feature maps, got {len(result)}"
                t.want = "length 5"
                t.got = f"length {len(result)}"
            cases.append(t)

            # check each element is a 4D tensor of shape (1, 1, 224, 224)
            for idx, fmap in enumerate(result):
                t = test_case()
                if not isinstance(fmap, torch.Tensor):
                    t.failed = True
                    t.msg = f"element {idx} is not a torch.Tensor"
                    t.want = torch.Tensor
                    t.got = type(fmap)
                elif fmap.dim() != 4:
                    t.failed = True
                    t.msg = f"tensor at index {idx} must be 4D (B,C,H,W)"
                    t.want = "4D tensor"
                    t.got = f"{fmap.shape}"
                elif fmap.shape[0] != 1 or fmap.shape[1] != 1:
                    t.failed = True
                    t.msg = f"tensor at index {idx} must have shape (1,1,H,W)"
                    t.want = "(1,1,224,224)"
                    t.got = f"{fmap.shape}"
                elif fmap.shape[2:] != (224, 224):
                    t.failed = True
                    t.msg = f"tensor at index {idx} must be upsampled to 224×224"
                    t.want = "(…,…,224,224)"
                    t.got = f"{fmap.shape}"
                cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing feature_map_strip: {e}"
            t.want = "No exception"
            t.got = "Exception raised"
            cases.append(t)
        # endregion

        return cases

    cases = g()
    print_feedback(cases)

def exercise_3(learner_func):
    def g():
        cases = []

        # region === general checks ===
        from types import FunctionType
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "salience_map has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        # endregion

        # region === functional test with toy data ===
        try:
            import torch
            from torchvision.models import resnet50

            # build a tiny model and set eval mode
            device = torch.device("cpu")
            model = resnet50(weights=None).to(device).eval()

            # create a dummy image tensor (leaf, requires no grad initially)
            H, W = 64, 48
            img = torch.rand(1, 3, H, W)

            # call learner function (_disable_ plotting)
            heatmap = learner_func(model, img.clone(), class_idx=0)

            # check return type
            t = test_case()
            if not isinstance(heatmap, torch.Tensor):
                t.failed = True
                t.msg = "salience_map should return a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(heatmap)
            cases.append(t)

            # check dimensions
            t = test_case()
            if heatmap.dim() != 2:
                t.failed = True
                t.msg = f"salience_map output must be 2D, got {heatmap.dim()}D"
                t.want = "2D tensor"
                t.got = f"{heatmap.shape}"
            cases.append(t)

            # check shape matches input H,W
            t = test_case()
            if heatmap.shape != (H, W):
                t.failed = True
                t.msg = f"salience_map output shape must be ({H},{W})"
                t.want = f"({H},{W})"
                t.got = f"{heatmap.shape}"
            cases.append(t)

            # check range [0,1]
            t = test_case()
            mn, mx = float(heatmap.min()), float(heatmap.max())
            if mn < 0 or mx > 1:
                t.failed = True
                t.msg = f"values must be in [0,1], got min={mn:.3f}, max={mx:.3f}"
                t.want = "all values ∈ [0,1]"
                t.got = f"min={mn:.3f}, max={mx:.3f}"
            cases.append(t)

            # check no NaNs or infs
            t = test_case()
            if not torch.isfinite(heatmap).all():
                t.failed = True
                t.msg = "salience_map contains NaN or infinite values"
                t.want = "all finite"
                t.got = "contains invalid"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing salience_map: {e}"
            t.want = "execution without exception"
            t.got = "exception raised"
            cases.append(t)
        # endregion

        return cases

    cases = g()
    print_feedback(cases)

def exercise_4(learner_func):
    def g():
        cases = []

        # region === general checks ===
        from types import FunctionType
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "simplified_cam has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        # endregion

        # region === functional tests with toy data ===
        try:
            import copy
            import torch
            from torchvision.models import resnet50

            device = torch.device("cpu")
            # build a ResNet-50 and set to eval
            base_model = resnet50(weights=None).to(device).eval()

            # dummy image tensor
            H, W = 32, 28
            img = torch.rand(1, 3, H, W, device=device)

            # 1) Test zero-weight case: FC weights zero => CAM must be all zeros
            model_zero = copy.deepcopy(base_model)
            with torch.no_grad():
                model_zero.fc.weight[:] = 0
                model_zero.fc.bias[:] = 0

            heat_zero = learner_func(model_zero, img.clone(), class_idx=0)
            t = test_case()
            if not isinstance(heat_zero, torch.Tensor):
                t.failed = True
                t.msg = "simplified_cam should return a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(heat_zero)
            elif heat_zero.shape != (H, W):
                t.failed = True
                t.msg = f"zero-weight CAM shape must be ({H},{W})"
                t.want = f"({H},{W})"
                t.got = f"{heat_zero.shape}"
            elif not torch.allclose(heat_zero, torch.zeros(H, W), atol=1e-6):
                t.failed = True
                t.msg = "zero-weight CAM must be all zeros"
                t.want = "all zeros"
                t.got = f"min={heat_zero.min():.3f}, max={heat_zero.max():.3f}"
            cases.append(t)

            # 2) Test shape and range on base model
            heat = learner_func(base_model, img.clone(), class_idx=0)
            # type check
            t = test_case()
            if not isinstance(heat, torch.Tensor):
                t.failed = True
                t.msg = "simplified_cam should return a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(heat)
            cases.append(t)

            # dimension check
            t = test_case()
            if heat.dim() != 2:
                t.failed = True
                t.msg = f"CAM output must be 2D, got {heat.dim()}D"
                t.want = "2D tensor"
                t.got = f"{heat.shape}"
            cases.append(t)

            # shape check
            t = test_case()
            if heat.shape != (H, W):
                t.failed = True
                t.msg = f"CAM output shape must be ({H},{W})"
                t.want = f"({H},{W})"
                t.got = f"{heat.shape}"
            cases.append(t)

            # range check [0,1]
            t = test_case()
            mn, mx = float(heat.min()), float(heat.max())
            if mn < 0 or mx > 1:
                t.failed = True
                t.msg = f"CAM values must be ∈ [0,1], got min={mn:.3f}, max={mx:.3f}"
                t.want = "all values ∈ [0,1]"
                t.got = f"min={mn:.3f}, max={mx:.3f}"
            cases.append(t)

            # finite check
            t = test_case()
            if not torch.isfinite(heat).all():
                t.failed = True
                t.msg = "CAM contains NaN or infinite values"
                t.want = "all finite"
                t.got = "contains invalid"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing simplified_cam: {e}"
            t.want = "execution without exception"
            t.got = "exception raised"
            cases.append(t)
        # endregion

        return cases

    cases = g()
    print_feedback(cases)