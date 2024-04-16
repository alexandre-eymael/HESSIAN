"""
This module provides functionalities for parsing command-line
arguments to configure HESSIAN experiments.

Functions:
    - get_args_parser: Get the argument parser for configuring HESSIAN experiments.
"""

import argparse

def get_args_parser():
    """
    Get the argument parser for configuring HESSIAN experiments.

    Returns:
        ArgumentParser: An ArgumentParser object configured with the following options:
            - --model_size: Size of the model (choices: "small", "base", "large").
            - --save_path: Path to save the checkpoints.
            - --save_freq: Save frequency.
            - --data_path: Path to the data.
            - --img_size: Size of the images.
            - --epochs: Number of epochs.
            - --optimizer: Optimizer.
            - --lr: Learning rate.
            - --train_prop: Proportion of the data used for training.
            - --batch_size: Batch size.
            - --max_samples: Maximum number of samples to use.
            - --seed: Seed for the random number generator.
            - --device: Device to use to train the model.
            - --load_all_in_ram: Load all the data in RAM.
            - --wandb_mode: Wandb mode (choices: "disabled", "online", "offline").
    """
    parser = argparse.ArgumentParser("HESSIAN", add_help=False)

    # Architecture
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "base", "large"], help="Size of the model")

    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./weights",
                        help="Path to save the checkpoints")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")

    # Dataset
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the images")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_prop", type=float, default=0.80,
                        help="Proportion of the data used for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use to train the model")
    parser.add_argument("--load_all_in_ram", action="store_true", help="Load all the data in RAM")

    parser.add_argument("--vertex_ai", action="store_true", help="Whether training is currently running on Vertex AI")
    # Misc
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["disabled", "online", "offline"], help="Wandb mode")

    return parser
