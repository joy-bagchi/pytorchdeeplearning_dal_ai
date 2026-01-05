import math
import os
from types import FunctionType
from unittest.mock import MagicMock, patch

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.models import ResNet

from dlai_grader.grading import print_feedback, test_case
import unittests_utils



def exercise_1(learner_class):
    def g():
        cases = []
        class_name = "ChestXRayDataModule"
        tmp_data_dir = None

        # ############################ Test case 1: Check if the learner's submission is a class ############################
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]

        t = test_case()
        if not issubclass(learner_class, pl.LightningDataModule):
            t.failed = True
            t.msg = f"{class_name} must inherit from pl.LightningDataModule"
            t.want = pl.LightningDataModule
            t.got = learner_class.__base__
            return [t]
        # ############################

        try:
            # Setup: Create dummy data for testing
            tmp_data_dir = unittests_utils.setup_dummy_data_dir()
            test_batch_size = 4
            dm = learner_class(data_dir=tmp_data_dir, batch_size=test_batch_size)

            # ############################ Test case 2: Test instantiation and __init__ method ############################
            
            # Test 2.1: Check attribute initialization (data_dir)
            t = test_case()
            if not hasattr(dm, 'data_dir') or dm.data_dir != tmp_data_dir:
                t.failed = True
                t.msg = "The `self.data_dir` attribute was not initialized correctly in `__init__`"
                t.want = "self.data_dir=data_dir"
                t.got = f"self.data_dir={getattr(dm, 'data_dir', 'Attribute not found')}"
                return [t]

            # Test 2.2: Check attribute initialization (batch_size)
            t = test_case()
            if not hasattr(dm, 'batch_size') or dm.batch_size != test_batch_size:
                t.failed = True
                t.msg = "The `self.batch_size` attribute was not initialized correctly in `__init__`"
                t.want = "self.batch_size=batch_size"
                t.got = f"self.batch_size={getattr(dm, 'batch_size', 'Attribute not found')}"
                return [t]

            # Test 2.3: Check attribute initialization (transformatons)

            # Test 2.3.1: Check if transforms are of the correct type
            t = test_case()
            if not (hasattr(dm, 'train_transform') and isinstance(dm.train_transform, transforms.Compose)) or \
               not (hasattr(dm, 'val_transform') and isinstance(dm.val_transform, transforms.Compose)):
                t.failed = True
                t.msg = "The `self.train_transform` and `self.val_transform` attributes must be `transforms.Compose` objects"
                t.want = "torchvision.transforms.Compose"
                t.got = f"train_transform is {type(getattr(dm, 'train_transform', None))}, val_transform is {type(getattr(dm, 'val_transform', None))}"
                return [t]
            
            # Test 2.3.2: Check the number of transformations in the training pipeline
            t = test_case()
            expected_train_transforms = 7
            num_train_transforms = len(dm.train_transform.transforms)
            if num_train_transforms != expected_train_transforms:
                t.failed = True
                t.msg = "The training transformation pipeline has an incorrect number of transforms. Make sure you are using TRAIN_TRANSFORM"
                t.want = f"{expected_train_transforms} transformations"
                t.got = f"{num_train_transforms} transformations"
                return [t]

            # Test 2.3.2: Check the number of transformations in the validation pipeline
            t = test_case()
            expected_val_transforms = 3
            num_val_transforms = len(dm.val_transform.transforms)
            if num_val_transforms != expected_val_transforms:
                t.failed = True
                t.msg = "The validation transformation pipeline has an incorrect number of transforms. Make sure you are using VAL_TRANSFORM"
                t.want = f"{expected_val_transforms} transformations"
                t.got = f"{num_val_transforms} transformations"
                return [t]

             # # ############################ Test case 3: Test the setup method ############################
            try:
                dm.setup()
            except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"The `setup` method failed to execute. Error: {e}"
                return cases + [t]
            
            # Test case 3.1: Check the path of the training dataset
            t = test_case()
            expected_train_path = os.path.join(dm.data_dir, "train")
            if not hasattr(dm, 'train_dataset') or dm.train_dataset.root != expected_train_path:
                t.failed = True
                t.msg = "The `train_dataset` was not created using the correct directory path"
                t.want = "Train dataset created from path: train_path"
                
                if not hasattr(dm, 'train_dataset'):
                    t.got = "The train_dataset attribute was not created."
                else:
                    # Get the actual path, which is 'tmp_data/val'
                    actual_path = getattr(dm.train_dataset, 'root', 'N/A')
                    # Extract only the subdirectory name ('val') from the full path
                    subdir = os.path.basename(actual_path)
                    t.got = f"A dataset created from path: {subdir}"
                return [t]

            # Test case 3.2: Check the path of the validation dataset
            t = test_case()
            expected_val_path = os.path.join(dm.data_dir, "val")
            if not hasattr(dm, 'val_dataset') or dm.val_dataset.root != expected_val_path:
                t.failed = True
                t.msg = "The `val_dataset` was not created using the correct directory path"
                t.want = "Validation dataset created from path: val_path"
                
                if not hasattr(dm, 'val_dataset'):
                    t.got = "The val_dataset attribute was not created."
                else:
                    # Get the actual path
                    actual_path = getattr(dm.val_dataset, 'root', 'N/A')
                    # Extract only the subdirectory name
                    subdir = os.path.basename(actual_path)
                    t.got = f"A dataset created from path: {subdir}"
                return [t]

            # Test case 3.4: Check the transformations of the training dataset
            t = test_case()
            expected_train_transforms = 7
            num_transforms = len(dm.train_dataset.transform.transforms)
            if num_transforms != expected_train_transforms:
                t.failed = True
                t.msg = "The `train_dataset` was configured with the wrong transformations. Make sure to use 'self.train_transform' for train set transformations"
                t.want = f"{expected_train_transforms} transformations"
                t.got = f"{num_transforms} transformations"
                return [t]

            # Test case 3.4: Check the transformations of the validation dataset
            t = test_case()
            expected_val_transforms = 3
            num_transforms = len(dm.val_dataset.transform.transforms)
            if num_transforms != expected_val_transforms:
                t.failed = True
                t.msg = "The `val_dataset` was configured with the wrong transformations. Make sure to use 'self.val_transform' for val set transformations"
                t.want = f"{expected_val_transforms} transformations"
                t.got = f"{num_transforms} transformations"
                return [t]

            # ############################ Test case 4: Test dataloader methods ############################
            try:
                train_loader = dm.train_dataloader()
                val_loader = dm.val_dataloader()
            except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"The `train_dataloader` and `val_dataloader` methods failed to execute. Error: {e}"
                return cases + [t]

            # Test case 4.1: Check the total number of images in the train dataloader's dataset
            t = test_case()
            train_loader = dm.train_dataloader()
            expected_num_images = 3
            actual_num_images = len(train_loader.dataset)
            if actual_num_images != expected_num_images:
                t.failed = True
                t.msg = "The train dataloader is not using a dataset with the correct number of images. Be sure to use the correct dataset and 'is_train_loader' flag"
                t.want = "Usage of `self.train_dataset` and `is_train_loader=True`"
                t.got = "Incorrect dataset and/or incorrect usage of `is_train_loader` flag."
            cases.append(t)

            # Test case 4.2: Check the total number of images in the val dataloader's dataset
            t = test_case()
            val_loader = dm.val_dataloader()
            expected_num_images = 2
            actual_num_images = len(val_loader.dataset)
            if actual_num_images != expected_num_images:
                t.failed = True
                t.msg = "The validation dataloader is not using a dataset with the correct number of images. Be sure to use the correct dataset and 'is_train_loader' flag"
                t.want = "Usage of `self.val_dataset` and `is_train_loader=False`"
                t.got = "Incorrect dataset and/or incorrect usage of `is_train_loader` flag."
            cases.append(t)
            
            # Test case 4.3: Check the batch size of the train dataloader
            t = test_case()
            if train_loader.batch_size != dm.batch_size:
                t.failed = True
                t.msg = "The train dataloader is not configured with the correct batch size. Make sure to pass `self.batch_size`"
                t.want = "self.batch_size"
                t.got = f"{train_loader.batch_size}"
            cases.append(t)

            # Test case 4.4: Check the batch size of the validation dataloader
            t = test_case()
            if val_loader.batch_size != dm.batch_size:
                t.failed = True
                t.msg = "The validation dataloaloader is not configured with the correct batch size. Make sure to pass `self.batch_size`"
                t.want = "self.batch_size"
                t.got = f"{val_loader.batch_size}"
            cases.append(t)
            # ############################
        
        except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"Failed to instantiate your {class_name} class. Error: {e}"
                return [t]
            
            
        finally:
            # Cleanup the dummy directory
            if tmp_data_dir:
                unittests_utils.cleanup_dummy_data_dir(tmp_data_dir)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_class):
    def g():
        cases = []
        class_name = "ChestXRayClassifier"

        # ############################ Test case 1: Check if the learner's submission is a class ############################
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]

        t = test_case()
        if not issubclass(learner_class, pl.LightningModule):
            t.failed = True
            t.msg = f"{class_name} must inherit from pl.LightningModule"
            t.want = pl.LightningDataModule
            t.got = learner_class.__base__
            return [t]
        # ############################
        
        try:
            # Setup for subsequent tests
            test_num_classes = 3
            dummy_weights_path = unittests_utils.setup_dummy_weights(num_classes=test_num_classes)
            model = learner_class(model_weights_path=dummy_weights_path, num_classes=test_num_classes)

            # ############################ Test case 2: Test instantiation and __init__ method ############################
            
            # Test 2.1.1: Check if self.model is the correct type
            t = test_case()
            if not isinstance(model.model, ResNet):
                t.failed = True
                t.msg = "The `self.model` attribute is not the correct model type. Be sure to use the `load_resnet18` helper function"
                t.want = "torchvision.models.ResNet"
                t.got = f"{type(model.model)}"
                return [t]

            # Test 2.1.2: Check if the model's final layer and weights are configured correctly
            t = test_case()
            # First, check if the number of output classes is correct
            if model.model.fc.out_features != test_num_classes:
                t.failed = True
                t.msg = "The model's final layer is not configured with the correct number of classes"
                t.want = f"Final layer configured for {test_num_classes} classes"
                t.got = f"Final layer configured for {model.model.fc.out_features} classes"
                return [t]
            else:
                # If the classes are correct, now check the weight configuration
                incorrectly_trainable_layers = []
                for name, param in model.model.named_parameters():
                    # Check for any layer outside of 'fc' that is trainable
                    if not name.startswith('fc.') and param.requires_grad:
                        incorrectly_trainable_layers.append(name)
                
                if incorrectly_trainable_layers:
                    t.failed = True
                    t.msg = "The model's weights are not configured correctly. All layers except the final 'fc' layer should be frozen"
                    t.want = "Only 'fc' layer parameters to be trainable"
                    t.got = f"Unexpected trainable layers found: {incorrectly_trainable_layers}"
                    return [t]

            # Test 2.2: Check for CrossEntropyLoss
            t = test_case()
            if not isinstance(model.loss_fn, nn.CrossEntropyLoss):
                t.failed = True
                t.msg = "self.loss_fn should be an instance of `nn.CrossEntropyLoss`"
                t.want = "An instance of `torch.nn.CrossEntropyLoss`"
                t.got = f"An instance of `{type(getattr(model, 'loss_fn', None))}`"
                return [t]

            # Test 2.3: Check for Accuracy
            t = test_case()
            # Check if the object is the correct specialized type and has the right number of classes.
            if not isinstance(model.accuracy, MulticlassAccuracy) or model.accuracy.num_classes != test_num_classes:
                t.failed = True
                t.msg = "self.accuracy metric is not configured correctly. It should be `Accuracy(task='multiclass', num_classes=...)`"
                t.want = f"An Accuracy metric with num_classes={test_num_classes}"
                
                acc_metric = getattr(model, 'accuracy', None)
                if acc_metric is None:
                    t.got = "The 'self.accuracy' attribute was not found."
                else:
                    actual_num_classes = getattr(acc_metric, 'num_classes', 'N/A')
                    if type(acc_metric).__name__ == 'MulticlassAccuracy':
                        t.got = f"An Accuracy metric with num_classes={actual_num_classes}"
                    else:
                        t.got = f"A `{type(acc_metric).__name__}` metric with num_classes={actual_num_classes}"
                return [t]

            # ############################ Test case 3: training_step & validation_step methods ############################
            
            # Test Case 3.1: training_step method
            t = test_case()
            dummy_x = torch.randn(2, 3, 224, 224)
            dummy_y = torch.randint(0, test_num_classes, (2,))
            loss = model.training_step((dummy_x, dummy_y))
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.grad_fn is None:
                t.failed = True
                t.msg = "The `training_step` method did not return a valid loss tensor"
                t.want = "A single-number (scalar) torch.Tensor with a gradient function, which is the direct output of your loss function."
                t.got = f"An object of type {type(loss)} with {loss.ndim} dimensions."
                return [t]

            
            # Test Case 3.2: validation_step method
            t = test_case()
            # Temporarily replace log_dict with a mock to inspect its inputs
            model.log_dict = MagicMock() 
            # Run the validation step with a dummy batch
            model.validation_step((dummy_x, dummy_y))
            # Check if log_dict was called
            if not model.log_dict.called:
                t.failed = True
                t.msg = "The `validation_step` did not seem to log any metrics"
                t.want = "A call to `self.log_dict` with calculated loss and accuracy."
                t.got = "The `self.log_dict` method was not called."
            else:
                # Get the dictionary that was passed to log_dict
                logged_dict = model.log_dict.call_args[0][0]
                val_loss = logged_dict.get('val_loss')
                val_acc = logged_dict.get('val_acc')
            
                # Check if a valid loss tensor was logged
                if val_loss is None or not isinstance(val_loss, torch.Tensor) or val_loss.ndim != 0:
                    t.failed = True
                    t.msg = "A valid validation loss was not calculated and logged in `validation_step`"
                    t.want = "A scalar torch.Tensor for 'val_loss'."
                    t.got = f"The value for 'val_loss' was {val_loss}."
                    return [t]
                # Check if a valid accuracy tensor was logged
                elif val_acc is None or not isinstance(val_acc, torch.Tensor) or val_acc.ndim != 0:
                    t.failed = True
                    t.msg = "A valid validation accuracy was not calculated and logged. Make sure to call `self.accuracy`"
                    t.want = "A scalar torch.Tensor for 'val_acc'."
                    t.got = f"The value for 'val_acc' was {val_acc}."
                    return [t]

            # ############################ Test case 4: configure_optimizers method ############################

            # Test Case 4.1: configure_optimizers method
            t = test_case()
            config = model.configure_optimizers()
            # Get the optimizer and scheduler from the returned dictionary
            optimizer = config.get("optimizer")
            scheduler = config.get("lr_scheduler", {}).get("scheduler")
            # Check if the optimizer and scheduler are of the correct types
            if not isinstance(optimizer, AdamW) or not isinstance(scheduler, ReduceLROnPlateau):
                t.failed = True
                t.msg = "The optimizer or scheduler is of the wrong type. Ensure you are calling the `define_optimizer_and_scheduler` helper function"
                t.want = "An `AdamW` optimizer and a `ReduceLROnPlateau` scheduler."
                opt_type = type(optimizer).__name__
                sched_type = type(scheduler).__name__
                t.got = f"Optimizer: {opt_type}, Scheduler: {sched_type}"
                return [t]
            
            # Test Case 4.2: Check for correct learning_rate and weight_decay usage
            t = test_case()
            # Define non-default values to test parameter passing
            test_lr = 0.5
            test_wd = 0.1
            # Instantiate a new model with these specific hyperparameters
            param_model = learner_class(
                model_weights_path=dummy_weights_path,
                learning_rate=test_lr,
                weight_decay=test_wd
            )
            # Get the configured optimizer
            param_config = param_model.configure_optimizers()
            param_optimizer = param_config.get("optimizer")
            # Get the actual values from the optimizer's parameter group
            actual_lr = param_optimizer.param_groups[0]['lr']
            actual_wd = param_optimizer.param_groups[0]['weight_decay']
            
            # Check if the actual values match the test values
            if actual_lr != test_lr or actual_wd != test_wd:
                t.failed = True
                t.msg = "The learning_rate or weight_decay from self.hparams was not correctly passed to the optimizer"
                t.want = f"'learning_rate=self.hparams.learning_rate' and 'weight_decay=self.hparams.weight_decay'"
                t.got = f"Learning Rate: {actual_lr}, Weight Decay: {actual_wd}"
                # This will stop the test and report the failure correctly
            cases.append(t)
        
        except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"Failed to instantiate your {class_name} class. Error: {e}"
                return [t]
            
            
        finally:
            if dummy_weights_path:
                unittests_utils.cleanup_dummy_weights(dummy_weights_path)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        
        # Setup test parameters
        test_epochs = 21
        test_threshold = 0.95

        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "early_stopping has inccorect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            # Call the learner's function to get the callback object
            callback = learner_func(test_epochs, test_threshold)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Your function failed to execute. Error: {e}"
            return [t]

        # ############################ Test case 2: Check the return type ############################
        t = test_case()
        if not isinstance(callback, EarlyStopping):
            t.failed = True
            t.msg = "The function should return an EarlyStopping callback object"
            t.want = "An instance of `lightning.pytorch.callbacks.EarlyStopping`"
            t.got = f"An instance of `{type(callback)}`"
            return [t]
        
        # ############################ Test Case 3: Check the 'monitor' parameter ############################
        t = test_case()
        expected_monitor = "val_acc"
        if callback.monitor != expected_monitor:
            t.failed = True
            t.msg = "The `monitor` parameter is not set correctly"
            t.want = f"monitor='{expected_monitor}'"
            t.got = f"monitor='{getattr(callback, 'monitor', 'Not found')}'"
        cases.append(t)

        # ############################ Test Case 4: Check the 'stopping_threshold' parameter ############################
        t = test_case()
        if callback.stopping_threshold != test_threshold:
            t.failed = True
            t.msg = "The `stopping_threshold` parameter is not set correctly"
            t.want = "stopping_threshold=stop_threshold"
            t.got = f"stopping_threshold={getattr(callback, 'stopping_threshold', 'Not found')}"
        cases.append(t)

        # ############################ Test Case 5: Check the 'patience' parameter ############################
        t = test_case()
        expected_patience = int(test_epochs / 2)
        if callback.patience != expected_patience:
            t.failed = True
            t.msg = "The `patience` parameter is not calculated correctly. It should be half of the total epochs, and as an integer"
            t.want = "patience = half of `num_epochs`, and of type int"
            t.got = f"patience={getattr(callback, 'patience', 'Not found')}, of type {type(callback.patience)}"
        cases.append(t)

        # ############################ Test Case 6: Check the 'mode' parameter ############################
        t = test_case()
        expected_mode = "max"
        if callback.mode != expected_mode:
            t.failed = True
            t.msg = "The `mode` parameter is not set correctly. Since a higher accuracy is better, it should be 'max'"
            t.want = f"mode='{expected_mode}'"
            t.got = f"mode='{getattr(callback, 'mode', 'Not found')}'"
        cases.append(t)
            
        return cases

    cases = g()
    print_feedback(cases)
    
    
def exercise_4(learner_func, data_module, classifier):
    def g():
        cases = []
        
        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "run_training has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        # ############################ Test Case 2: Check return types from a dry run ############################
        t = test_case()
        try:
            # Instantiate real modules for this test
            dummy_dm = data_module(data_dir="./chest_xray/", batch_size=2)
            dummy_model = classifier(model_weights_path="./resnet18_chest_xray_classifier_weights.pth")
            dummy_callback = MagicMock(spec=pl.Callback)

            result_tuple_real = learner_func(
                model=dummy_model,
                data_module=dummy_dm,
                num_epochs=1,
                callback=dummy_callback,
                progress_bar=False,
                dry_run=True
            )

            # Test Case 2.1: Return is a tuple
            if not isinstance(result_tuple_real, tuple) or len(result_tuple_real) != 2:
                t.failed = True
                t.msg = "The function should return a tuple with two items (trainer, model)"
                t.want = "A tuple of length 2"
                if hasattr(result_tuple_real, '__len__'):
                    t.got = f"{len(result_tuple_real)} items were returned."
                else:
                    t.got = "An object that is not a tuple was returned."
                return [t]

            # Test Case 2.2: Return types
            trainer_obj, model_obj = result_tuple_real
            if not isinstance(trainer_obj, pl.Trainer) or not isinstance(model_obj, pl.LightningModule):
                t.failed = True
                t.msg = "The function did not return the correct object types"
                t.want = "A tuple of (pl.Trainer, pl.LightningModule)"
                if type(model_obj).__name__ == 'ChestXRayClassifier':
                    t.got = f"A tuple containing a `{type(trainer_obj).__name__}` and a `LightningModule`."
                else:
                    t.got = f"A tuple containing a `{type(trainer_obj).__name__}` and a `{type(model_obj).__name__}`."
                return [t]

        except Exception as e:
            t.failed = True
            t.msg = f"Your function failed to execute during the test. Error: {e}"
            return [t]
                
        # --- Mocking-based Tests for Parameter Checking ---
        mock_model = MagicMock(spec=pl.LightningModule)
        mock_data_module = MagicMock(spec=pl.LightningDataModule)
        mock_callback = MagicMock(spec=pl.Callback)
        test_epochs = 10
        
        with patch('lightning.pytorch.Trainer') as mock_trainer_class:
            mock_trainer_instance = mock_trainer_class.return_value
            try:
                learner_func(
                    model=mock_model,
                    data_module=mock_data_module,
                    num_epochs=test_epochs,
                    callback=mock_callback
                )
            except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"Your function failed to execute during the test. Error: {e}"
                return [t]
            
            # ############################ Test Case 3: Check that pl.Trainer was instantiated ############################
            t = test_case()
            try:
                mock_trainer_class.assert_called_once()
            except AssertionError:
                t.failed = True
                t.msg = "The `pl.Trainer` was not instantiated inside your function"
                t.want = "A call to `pl.Trainer(...)`"
                t.got = "No call was detected."
                return [t]
            
            trainer_kwargs = mock_trainer_class.call_args.kwargs
            
            # ############################ Test Case 4: Check the 'callbacks' parameter ############################
            actual_callbacks = trainer_kwargs.get('callbacks')
            t = test_case()
            if trainer_kwargs.get('callbacks') != [mock_callback]:
                t.failed = True
                t.msg = "The `pl.Trainer` was not instantiated with the correct callback. Remember to pass it as a list"
                t.want = "callbacks=[callback]"
                if not isinstance(actual_callbacks, list):
                    t.got = "The callback was not passed inside a list."
                else:
                    t.got = "An incorrect or empty list was passed for the callbacks."
                return [t]
            
            # ############################ Test Case 5: Check that trainer.fit() was called correctly ############################
            t = test_case()
            try:
                mock_trainer_instance.fit.assert_called_once_with(mock_model, mock_data_module)
            except AssertionError:
                t.failed = True
                t.msg = "The `trainer.fit()` method was not called with the correct arguments"
                t.want = "A call to `trainer.fit(model, data_module)`"
                if not mock_trainer_instance.fit.called:
                    t.got = "The .fit() method was not called at all."
                else:
                    # Get the types of the arguments that were actually passed
                    actual_args = mock_trainer_instance.fit.call_args.args
                    arg_types = [type(arg).__name__ for arg in actual_args]
                    t.got = f"The .fit() method was called with arguments of types: {arg_types}"
                return [t]
            
            # ############################ Test Case 6: Check the 'max_epochs' parameter ############################
            t = test_case()
            if trainer_kwargs.get('max_epochs') != test_epochs:
                t.failed = True
                t.msg = "The `pl.Trainer` was not instantiated with the correct `max_epochs`"
                t.want = "max_epochs=num_epochs"
                t.got = f"max_epochs={trainer_kwargs.get('max_epochs')}"
            cases.append(t)

            # ############################ Test Case 7: Check the 'precision' parameter ############################
            t = test_case()
            if trainer_kwargs.get('precision') != '16-mixed':
                t.failed = True
                t.msg = "The `pl.Trainer` was not configured for mixed-precision training"
                t.want = "precision='16-mixed'"
                t.got = f"precision='{trainer_kwargs.get('precision')}'"
            cases.append(t)

            
        return cases

    cases = g()
    print_feedback(cases)