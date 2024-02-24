import argparse

def get_args_parser():

    parser = argparse.ArgumentParser("HESSIAN", add_help=False)

    # Architecture
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "large"], help="Size of the model")

    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save the checkpoints")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")

    # Dataset
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the images")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_prop", type=float, default=0.80, help="Proportion of the data used for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use to train the model")
    parser.add_argument("--load_all_in_ram", action="store_true", help="Load all the data in RAM")

    # Misc
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["disabled", "online", "offline"], help="Wandb mode")

    return parser