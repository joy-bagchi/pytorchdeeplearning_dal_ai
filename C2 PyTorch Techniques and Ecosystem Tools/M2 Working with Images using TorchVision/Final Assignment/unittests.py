import inspect
import re
from types import FunctionType

from dlai_grader.grading import test_case, print_feedback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models
from torchvision.models import MobileNetV3

import unittests_utils
from unittests_utils import MockImageFolder



def exercise_1(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_dataset_splits has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        expected_type = ImageFolder
        dataset_path = "../../AIvsReal_sampled"
        train_path = dataset_path + "/train"
        val_path = dataset_path + "/test"

        learner_train, learner_val = learner_func(dataset_path)

        ### Return type check (train)
        t = test_case()
        if not isinstance(learner_train, expected_type):
            t.failed = True
            t.msg = "Incorrect train_dataset type returned from create_dataset_splits"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_train)}"
            return [t]

        ### Return type check (val)
        t = test_case()
        if not isinstance(learner_val, expected_type):
            t.failed = True
            t.msg = "Incorrect val_dataset type returned from create_dataset_splits"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_val)}"
            return [t]

        ### Root path check (train)
        t = test_case()
        if learner_train.root != train_path:
            t.failed = True
            t.msg = f"Incorrect root path for the train_dataset"
            t.want = f"root path as 'root={train_path}'"
            t.got = f"root path as 'root={learner_train.root}'"
        cases.append(t)

        ### Root path check (val)
        t = test_case()
        if learner_val.root != val_path:
            t.failed = True
            t.msg = f"Incorrect root path for the val_dataset"
            t.want = f"root path as 'root={val_path}'"
            t.got = f"root path as 'root={learner_val.root}'"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


    
def exercise_2(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "define_transformations has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        mean = torch.tensor([0.4, 0.4, 0.4])
        std = torch.tensor([0.3, 0.3, 0.3])
        expected_function_return = transforms.Compose
        expected_train_rand_resize_crop = (224, 224)
        expected_brightness = 0.2
        expected_contrast = 0.2
        
        learner_train_transform, learner_val_transform = learner_func(mean=mean, std=std)

        ### Return type check 1 (train_transform)
        t = test_case()
        if not isinstance(learner_train_transform, expected_function_return):
            t.failed = True
            t.msg = "Incorrect train_transform return type"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_train_transform)}"
            return [t]
        
        ### Return type check 2 (val_transform)
        t = test_case()
        if not isinstance(learner_val_transform, expected_function_return):
            t.failed = True
            t.msg = "Incorrect val_transform return type"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_val_transform)}"
            return [t]

        # Verify the number of train transformations, should be 5
        t = test_case()
        if len(learner_train_transform.transforms) != 5:
            t.failed = True
            t.msg = f"Expected 5 train transformations, but found {len(learner_train_transform.transforms)}"
            t.want = "5 train transformations in define_transformations. RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor and Normalize"
            t.got = f"{len(learner_train_transform.transforms)} train transformations"
            return [t]
        
        # Verify the number of val transformations, should be 3
        t = test_case()
        if len(learner_val_transform.transforms) != 3:
            t.failed = True
            t.msg = f"Expected 3 val transformations, but found {len(learner_val_transform.transforms)}"
            t.want = "3 val transformations in define_transformations. Resize, ToTensor and Normalize"
            t.got = f"{len(learner_val_transform.transforms)} val transformations"
            return [t]
        
        ###################################################################################
        
        ### Check for RandomResizedCrop in train_transform
        rand_resized_crop_found = False
        found_correct_size = False
        found_sizes = []
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomResizedCrop):
                rand_resized_crop_found = True
                found_sizes.append(transform.size)
                if transform.size == expected_train_rand_resize_crop:
                    found_correct_size = True
                    break  # Found the expected one, no need to continue

        t = test_case()
        if not rand_resized_crop_found:
            t.failed = True
            t.msg = "RandomResizedCrop transform not found in train_transform"
            t.want = "train_transform to include RandomResizedCrop transform"
            t.got = "train_transform without RandomResizedCrop transform"
        elif not found_correct_size:
            t.failed = True
            t.msg = f"RandomResizedCrop found in train_transform, but with incorrect size"
            t.want = f"{expected_train_rand_resize_crop}"
            t.got = f"{found_sizes[0] if found_sizes else 'No incorrect size found'}"
        cases.append(t)
        
        ### Check for RandomHorizontalFlip in train_transform
        hflip_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomHorizontalFlip):
                hflip_found = True
                break

        t = test_case()
        if hflip_found == False:
            t.failed = True
            t.msg = "RandomHorizontalFlip transform not found in train_transform"
            t.want = "RandomHorizontalFlip transform present in train_transform"
            t.got = "No RandomHorizontalFlip transform in train_transform"
        cases.append(t)
        
        ### Check for ColorJitter in train_transform with specific brightness and contrast
        color_jitter_found, found_correct_jitter, found_brightness_val, found_contrast_val = unittests_utils.check_color_jitter(
            learner_train_transform, expected_brightness, expected_contrast
        )

        t = test_case()
        if not color_jitter_found:
            t.failed = True
            t.msg = "ColorJitter transform not found in train_transform"
            t.want = "train_transform to include ColorJitter transform"
            t.got = "train_transform without ColorJitter transform"
        elif not found_correct_jitter:
            t.failed = True
            t.msg = "ColorJitter found in train_transform, but with incorrect brightness and/or contrast"
            t.want = f"(brightness={expected_brightness}, contrast={expected_contrast})"
            t.got = f"(brightness={found_brightness_val}, contrast={found_contrast_val})"
        cases.append(t)
        

        ### Check for ToTensor in train_transform
        totensor_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "ToTensor transform not found in train_transform"
            t.want = "ToTensor transform present in train_transform"
            t.got = "No ToTensor transform in train_transform"
        cases.append(t)
        
        
        ### Check for Normalize in train_transform with specific mean and std
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if torch.equal(found_mean, mean) and torch.equal(found_std, std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Normalize transform not found in train_transform"
            t.want = "train_transform to include Normalize transform"
            t.got = "train_transform without Normalize transform"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize found in train_transform, but with incorrect mean and/or std"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean.tolist()) if found_mean is not None else None}, std={tuple(found_std.tolist()) if found_std is not None else None})"
        cases.append(t)
            
        ###################################################################################
        
        expected_val_resize = (224, 224)
        
        ### Check for Resize in val_transform with specific size
        resize_found_val = False
        found_correct_resize_val = False
        found_resize_size_val = None
        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.Resize):
                resize_found_val = True
                found_resize_size_val = transform.size
                if transform.size == expected_val_resize:
                    found_correct_resize_val = True
                break

        t = test_case()
        if not resize_found_val:
            t.failed = True
            t.msg = "Resize transform not found in val_transform"
            t.want = "val_transform to include Resize transform"
            t.got = "val_transform without Resize transform"
        elif not found_correct_resize_val:
            t.failed = True
            t.msg = "Resize found in val_transform, but with incorrect pixel size"
            t.want = f"{expected_val_resize}"
            t.got = f"{found_resize_size_val}"
        cases.append(t)
        
        ### Check for ToTensor in val_transform
        totensor_found = False

        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "ToTensor transform not found in val_transform"
            t.want = "ToTensor transform present in val_transform"
            t.got = "No ToTensor transform in val_transform"
        cases.append(t)
        
        ### Check for Normalize in val_transform with specific mean and std
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if torch.equal(found_mean, mean) and torch.equal(found_std, std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Normalize transform not found in val_transform"
            t.want = "val_transform to include Normalize transform"
            t.got = "val_transform without Normalize transform"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize found in val_transform, but with incorrect mean and/or std"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean.tolist()) if found_mean is not None else None}, std={tuple(found_std.tolist()) if found_std is not None else None})"
        cases.append(t)
        
        ###################################################################################
        
        # Check the order of transformations (train_transform)
        expected_train_order = [
            transforms.RandomResizedCrop,
            transforms.RandomHorizontalFlip,
            transforms.ColorJitter,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_train_order = [type(transform) for transform in learner_train_transform.transforms]

        t = test_case()
        if learner_train_order != expected_train_order:
            t.failed = True
            t.msg = "Train transformations are not applied in the expected order"
            t.want = f"[{', '.join([t.__name__ for t in expected_train_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_train_order])}]"
        cases.append(t)
        
        # Check the order of transformations (val_transform)
        expected_val_order = [
            transforms.Resize,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_val_order = [type(transform) for transform in learner_val_transform.transforms]

        t = test_case()
        if learner_val_order != expected_val_order:
            t.failed = True
            t.msg = "Validation transformations are not applied in the expected order"
            t.want = f"[{', '.join([t.__name__ for t in expected_val_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_val_order])}]"
        cases.append(t)
        
        return cases

    cases = g()
    print_feedback(cases)


    
def exercise_3(learner_func):
    def g():
        
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_data_loaders has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        mock_train, mock_val = unittests_utils.generate_mock_datasets()
        learner_train_loader, learner_val_loader, learner_trainset, learner_valset = learner_func(mock_train, mock_val, batch_size=4)
        
        ### Return type check 1 (train_loader)
        t = test_case()
        if not isinstance(learner_train_loader, DataLoader):
            t.failed = True
            t.msg = "train_loader has incorrect return type"
            t.want = DataLoader
            t.got = type(learner_train_loader)
            return [t]
        
        ### Return type check 2 (val_loader)
        t = test_case()
        if not isinstance(learner_val_loader, DataLoader):
            t.failed = True
            t.msg = "val_loader has incorrect return type"
            t.want = DataLoader
            t.got = type(learner_val_loader)
            return [t]
        
        ### Return type check 3 (train_transform_dataset)
        t = test_case()
        # Check if it's an instance of EITHER the real or the mock class
        if not isinstance(learner_trainset, (ImageFolder, MockImageFolder)):
            t.failed = True
            t.msg = "trainset has incorrect return type."
            t.want = "A Dataset object like ImageFolder"
            t.got = type(learner_trainset)
            return [t]

        ### Return type check 4 (val_transform_dataset)
        t = test_case()
        # Check if it's an instance of EITHER the real or the mock class
        if not isinstance(learner_valset, (ImageFolder, MockImageFolder)):
            t.failed = True
            t.msg = "valset has incorrect return type."
            t.want = "A Dataset object like ImageFolder"
            t.got = type(learner_valset)
            return [t]
        
        learner_code = inspect.getsource(learner_func)
        cleaned_code = unittests_utils.remove_comments(learner_code)
        
        # Check for the mention of 'define_transformations' in the learner's code
        t = test_case()
        if "define_transformations" not in cleaned_code:
            t.failed = True
            t.msg = "create_data_loaders implementation does not seem to call 'define_transformations' function for the initialization of 'train_transform, val_transform'"
            t.want = "'define_transformations' function for the initialization of 'train_transform, val_transform'"
            t.got = "'train_transform, val_transform' being initialized some other way"
            return [t]
        
        # Check if the transform attribute was assigned to the trainset
        t = test_case()
        if getattr(learner_trainset, 'transform', None) is None:
            t.failed = True
            t.msg = "The `transform` attribute of the returned training set is either missing or was not assigned a value"
            t.want = "`trainset.transform` to be assigned the training transformations."
            t.got = "The `transform` attribute was either not found or was set to `None`."
            return [t]

        # Check if the transform attribute was assigned to the valset
        t = test_case()
        if getattr(learner_valset, 'transform', None) is None:
            t.failed = True
            t.msg = "The `transform` attribute of the returned validation set is either missing or was not assigned a value"
            t.want = "`valset.transform` to be assigned the validation transformations."
            t.got = "The `transform` attribute was either not found or was set to `None`."
            return [t]

        # Check if the correct number of transformations were applied to the trainset
        num_transforms = len(learner_trainset.transform.transforms)

        t = test_case()
        if num_transforms != 5:
            t.failed = True
            t.msg = "Incorrect transformations applied to trainset. Make sure are applying train_transform"
            t.want = "Expected 5 transformations to be applied."
            t.got = f"Found {num_transforms} transformations."
        cases.append(t)

        # Check if the correct number of transformations were applied to the valset
        num_transforms_val = len(learner_valset.transform.transforms)
        
        t = test_case()
        if num_transforms_val != 3:
            t.failed = True
            t.msg = "Incorrect transformations applied to valset. Make sure you are applying val_transform"
            t.want = "Expected 3 transformations to be applied."
            t.got = f"Found {num_transforms_val} transformations."
        cases.append(t)
        
        expected_train_shape = torch.Size([4, 3, 224, 224])
        expected_val_shape = torch.Size([4, 3, 224, 224])

        ### Check train_loader
        for batch_idx, (images, labels) in enumerate(learner_train_loader):
            if batch_idx == 2:
                learner_train_shape = images.shape
                break
                
        # Check for correct initialization of train_loader with train_transform_dataset
        t = test_case()
        if len(learner_train_loader) != 25:
            t.failed = True
            t.msg = "Incorrect length of train_loader. Please make sure you are using train_transform_dataset when setting up train_loader"
            t.want = "train_loader to use train_transform_dataset"
            t.got = "train_loader not using train_transform_dataset"
        cases.append(t)
        
        # Batch Size (train_loader)
        t = test_case()
        if expected_train_shape[0] != learner_train_shape[0]:
            t.failed = True
            t.msg = "Incorrect batch_size of train_loader"
            t.want = "batch_size=batch_size"
            t.got = f"batch_size={learner_train_shape[0]}"
        cases.append(t)

        ### Check val_loader
        for batch_idx, (images, labels) in enumerate(learner_val_loader):
            if batch_idx == 2:
                learner_val_shape = images.shape
                break
                
        # Check for correct initialization of val_loader with val_transform_dataset
        t = test_case()
        if len(learner_val_loader) != 19:
            t.failed = True
            t.msg = "Incorrect length of val_loader. Please make sure you are using val_transform_dataset when setting up val_loader"
            t.want = "val_loader to use val_transform_dataset"
            t.got = "val_loader not using val_transform_dataset"
        cases.append(t)
        
        # Batch Size (val_loader)
        t = test_case()
        if expected_val_shape[0] != learner_val_shape[0]:
            t.failed = True
            t.msg = "Incorrect batch_size of val_loader"
            t.want = "batch_size=batch_size"
            t.got = f"batch_size={learner_val_shape[0]}"
        cases.append(t)
        
        # Check shuffle (train_loader)
        t = test_case()
        if not unittests_utils.check_shuffle(learner_train_loader, True):
            t.failed = True
            t.msg = "Incorrect shuffle of train_loader. Please make sure you are setting the shuffle as True for the train_loader"
            t.want = "shuffle=True"
            t.got = "shuffle=False or shuffle=None"
        cases.append(t)
        
        # Check shuffle (val_loader)
        t = test_case()
        if not unittests_utils.check_shuffle(learner_val_loader, False):
            t.failed = True
            t.msg = "Incorrect shuffle of val_loader. Please make sure you are setting the shuffle as False for the val_loader"
            t.want = "shuffle=False"
            t.got = "shuffle=True or shuffle=None"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    


def exercise_4_1(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "load_mobilenetv3_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        local_weights = "./mobilenet_weights/mobilenet_v3_large-8738ca79.pth"
        learner_model = learner_func(local_weights)
        
        ### Return type check
        t = test_case()
        expected_type = MobileNetV3
        if not isinstance(learner_model, expected_type):
            t.failed = True
            t.msg = f"Incorrect model type returned. Make sure you are loading the model 'mobilenet_v3_large' as instructed"
            t.want = f"model to be of type {expected_type}"
            t.got = f"{type(learner_model)}"
            return [t]
        
        ### Check last classifier layer
        t = test_case()
        try:
            last_classifier = learner_model.classifier[-1]
            is_linear = isinstance(last_classifier, nn.Linear)
            has_correct_in_features = last_classifier.in_features == 1280
            has_correct_out_features = last_classifier.out_features == 1000

            if not (is_linear and has_correct_in_features and has_correct_out_features):
                t.failed = True
                t.msg = "The last model layer of the classifier does not match the expected Linear layer (in_features=1280, out_features=1000)."
                t.want = "Loaded model to be 'mobilenet_v3_large' as instructed"
                t.got = "Some model other than 'mobilenet_v3_large' being loaded"
                return [t]
        except Exception as e:
            t.failed = True
            t.msg = f"An error occurred while trying to access or check the last classifier layer: {e}"
            t.want = "Loaded modelto be 'mobilenet_v3_large' as instructed"
            t.got = "Some model other than 'mobilenet_v3_large' being loaded"
            return [t]
        
        # Check for the presence of 'weights=None'
        learner_code = inspect.getsource(learner_func)
        cleaned_code = unittests_utils.remove_comments(learner_code)
        pretrained_flag_pattern = r"weights\s*=\s*None"
        t = test_case()
        if not re.search(pretrained_flag_pattern, cleaned_code):
            t.failed = True
            t.msg = "The parameter 'weights' does not seem to be set as None"
            t.want = "weights=None"
            t.got = "Something else"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


    
def exercise_4_2(learner_func):
    def g():
        
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "update_model_last_layer has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        initial_model = tv_models.mobilenet_v3_large(weights=None)
        local_weights = "./mobilenet_weights/mobilenet_v3_large-8738ca79.pth"
        state_dict = torch.load(local_weights, map_location=torch.device('cpu'))
        initial_model.load_state_dict(state_dict)
        learner_output = learner_func(initial_model, num_classes=8)
        
        ### Return type check
        t = test_case()
        expected_type = MobileNetV3
        if not isinstance(learner_output, expected_type):
            t.failed = True
            t.msg = f"Incorrect model type returned"
            t.want = f"{expected_type}"
            t.got = f"{type(learner_output)}"
            return [t]
        
        ### Check if feature parameters are frozen
        t = test_case()
        all_frozen = True
        for name, param in learner_output.features.named_parameters():
            if param.requires_grad:
                all_frozen = False
                break
        if not all_frozen:
            t.failed = True
            t.msg = "Not all feature parameters are frozen (requires_grad is not False)"
            t.want = "'requires_grad = False' for all feature layers of the model"
            t.got = "'requires_grad' are not false for one or more of the feature layers of the model"
        cases.append(t)
        
        ### Check if the last classifier layer is a new nn.Linear layer
        t = test_case()
        if not isinstance(learner_output.classifier[-1], nn.Linear):
            t.failed = True
            t.msg = "The last layer of the classifier should be a new nn.Linear layer."
            t.want = nn.Linear
            t.got = type(learner_output.classifier[-1])
            return [t]
        
        ### Check the output features of the last linear layer
        t = test_case()
        last_layer = learner_output.classifier[-1]
        expected_out_features = 8
        if not hasattr(last_layer, 'out_features') or last_layer.out_features != expected_out_features:
            t.failed = True
            t.msg = f"The last linear layer did not match the expected output features"
            if hasattr(last_layer, 'in_features') and last_layer.in_features == 1280:
                t.want = "Linear(in_features=1280, out_features=num_classes, bias=True)"
                t.got = f"{last_layer}"
            else:
                t.want = "Linear(in_features=num_features, out_features=num_classes, bias=True)"
                t.got = f"{last_layer}"    
        cases.append(t)
        
        return cases

    cases = g()
    print_feedback(cases)