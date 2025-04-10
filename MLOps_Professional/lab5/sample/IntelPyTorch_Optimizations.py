#!/usr/bin/env python
# encoding: utf-8

"""
==============================================================
 Copyright © 2022 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
"""

import os
from time import time
import torch
import torchvision
import intel_extension_for_pytorch as ipex
import argparse
from safetensors.torch import save_file

# Hyperparameters and constants
SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab5")
LR = 0.001
MOMENTUM = 0.9
DOWNLOAD = True
DATA = "datasets/cifar10/"

"""
Function to run a test case
"""


def trainModel(
    train_loader: torch.utils.data.DataLoader,
    modelName: str = "myModel",
    dtype: str = "fp32",
) -> float:
    """
    Trains a ResNet50 model using the specified data type and data loader.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader object containing the training data.
        modelName (str): The name of the model.
        dtype (str): The data type for model parameters, supported values - 'fp32', 'bf16'.

    Returns:
        float: The time in seconds it takes to train the model.
    """

    # Initialize the model
    model = torchvision.models.resnet50()
    model = model.to(memory_format=torch.channels_last)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()

    # Optimize with BF16 or FP32 (default)
    if "bf16" == dtype:
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=torch.bfloat16
        )
    else:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Train the model
    num_batches = len(train_loader)
    start_time = time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if "bf16" == dtype:
            with torch.cpu.amp.autocast():  # Auto Mixed Precision
                # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
                data = data.to(memory_format=torch.channels_last)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
        else:
            # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
            data = data.to(memory_format=torch.channels_last)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
        optimizer.step()
        if 0 == (batch_idx + 1) % 50:
            print("Batch %d/%d complete" % (batch_idx + 1, num_batches))
    end_time = time()
    training_time = end_time - start_time
    print("Training took %.3f seconds" % (training_time))

    # Save a checkpoint of the trained model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    checkpoint_filename = f"checkpoint_{modelName}.safetensors"
    checkpoint_path = os.path.normpath(os.path.join(SAFE_BASE_DIR, checkpoint_filename))
    if not checkpoint_path.startswith(SAFE_BASE_DIR):
        raise ValueError("Path is not within the allowed model directory.")
    try:
        save_file(checkpoint, checkpoint_path)
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        raise

    return training_time


"""
Perform all types of training in main function
"""


def main(FLAGS: argparse.Namespace) -> None:
    """
    Main function to perform all types of training.

    Args:
        FLAGS (argparse.Namespace): Parsed command-line arguments.
    """
    # Check if hardware supports AMX
    import sys

    sys.path.append("../../")
    from cpuinfo import get_cpu_info

    info = get_cpu_info()
    flags = info["flags"]
    amx_supported = False
    for flag in flags:
        if "amx" in flag:
            amx_supported = True
            break
    if not amx_supported:
        print("AMX is not supported on current hardware. Code sample cannot be run.\n")
        return

    # Load dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=FLAGS.batch_size
    )

    # Train models and acquire training times
    print(f"Training model with {FLAGS.data_type}")
    try:
        training_time = trainModel(
            train_loader, modelName=f"{FLAGS.data_type}", dtype=f"{FLAGS.data_type}"
        )
        print("Summary")
        print("training time: %.3f" % training_time)
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    """
    Main entry point for the script.

    This block parses command-line arguments and calls the main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dtype",
        "--data_type",
        type=str,
        default="fp32",
        help="pytorch data type options available are fp32 and bf16",
    )
    parser.add_argument(
        "-batch", "--batch_size", type=int, default=128, help="set training batch size"
    )
    FLAGS = parser.parse_args()

    # Validate inputs
    if FLAGS.data_type not in ["fp32", "bf16"]:
        raise ValueError("Invalid data type. Supported values are 'fp32' and 'bf16'.")
    if not isinstance(FLAGS.batch_size, int) or FLAGS.batch_size <= 0:
        raise ValueError("Invalid batch size. It should be a positive integer.")

    try:
        main(FLAGS)
        print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
