"""
Module: WandbLogger
Description: This module provides a class for logging metrics, images,
and other data to Weights and Biases (W&B) platform.

Functions:
    - __init__: Initializes the W&B logger with the specified configuration.
    - log: Logs the provided metrics to W&B.
    - make_image: Creates an image object to be logged to W&B.
    - create_table: Creates a table object to be logged to W&B.
    - log_roc_curve: Logs the ROC curve to W&B.
    - add_html_to_table: Adds HTML content to a table to be logged to W&B.
    - summary: Sets a summary value in the W&B run.
    - get_summary: Retrieves a summary value from the W&B run.
    - finish: Finishes the W&B run.
    - get_name: Retrieves the name of the W&B run.
    - start_watch: Starts watching the specified model for logging to W&B.
    - get_config: Retrieves the configuration from W&B.

Dependencies: wandb
"""

import wandb

class WandbLogger:
    """
    A wrapper class for logging experiments using the Weights & Biases (W&B) platform.

    This class provides methods to log metrics, images, tables, and other data to the W&B dashboard,
    as well as methods for managing experiment summaries and configurations.

    Attributes:
        config (dict): A dictionary containing experiment configuration parameters.
        mode (str): The mode of W&B usage, either "online" or "offline".

    Methods:
        __init__(config, mode="online"): Initializes the WandbLogger with the given configuration
            and mode.
        log(metrics): Logs the provided metrics to the W&B dashboard.
        make_image(image, caption):
        Creates a W&B image object with the given image data and caption.
        create_table(columns): Creates a W&B table with the specified columns.
        log_roc_curve(gts, preds): Logs a ROC curve to the W&B dashboard using the ground truth
            and predicted values.
        add_html_to_table(table, file_path):
        Adds HTML data from the specified file to the provided table.
        summary(key, value): Sets a summary value for the specified key in the experiment.
        get_summary(key): Retrieves the summary value for the specified key from the experiment.
        finish(): Finalizes the experiment and closes the W&B logging session.
        get_name(): Retrieves the name of the current W&B run.
        start_watch(model, log_freq=5): Starts watching the specified PyTorch model for logging
            gradients and parameters updates.
        get_config(): Retrieves the configuration settings for the current experiment.
    """
    def __init__(self, config, mode="online"):
        """
        Initializes the WandbLogger with the specified configuration.

        Args:
            config (dict): Configuration parameters for the W&B run.
            mode (str, optional): Mode for the W&B run, either "online" or "offline".
        """
        wandb.init(entity = "lba_mlops", project="mlops", config=config, mode=mode)

    def log(self, metrics):
        """
        Logs the provided metrics to W&B.

        Args:
            metrics (dict): Dictionary containing the metrics to be logged.
        """
        wandb.log(metrics)

    def make_image(self, image, caption):
        """
        Creates an image object to be logged to W&B.

        Args:
            image: The image data to be logged.
            caption (str): Caption for the image.

        Returns:
            wandb.Image: An image object compatible with W&B.
        """
        return wandb.Image(image, caption=caption)

    def create_table(self, columns):
        """
        Creates a table object to be logged to W&B.

        Args:
            columns (list): List of column names for the table.

        Returns:
            wandb.Table: A table object compatible with W&B.
        """
        return wandb.Table(columns = columns)

    def log_roc_curve(self, gts, preds):
        """
        Logs the ROC curve to W&B.

        Args:
            gts (list): Ground truth labels.
            preds (list): Predicted probabilities.
        """
        wandb.log({"roc": wandb.plot.roc_curve(y_true=gts, y_probas=preds,
                                               labels=["healthy"], classes_to_plot=[0])})

    def add_html_to_table(self, table, file_path):
        """
        Adds HTML content to a table to be logged to W&B.

        Args:
            table (wandb.Table): Table object to add HTML content.
            file_path (str): Path to the HTML file.
        """
        table.add_data(wandb.Html(file_path))

    def summary(self, key, value):
        """
        Sets a summary value in the W&B run.

        Args:
            key (str): Key for the summary value.
            value: Value to be stored.
        """
        wandb.run.summary[key] = value

    def get_summary(self, key):
        """
        Retrieves a summary value from the W&B run.

        Args:
            key (str): Key of the summary value to retrieve.

        Returns:
            Value: The summary value.
        """
        return wandb.run.summary[key]

    def finish(self):
        """Finishes the W&B run."""
        wandb.finish()

    def get_name(self):
        """
        Retrieves the name of the W&B run.

        Returns:
            str: The name of the W&B run.
        """
        return wandb.run.name

    def start_watch(self, model, log_freq=5):
        """
        Starts watching the specified model for logging to W&B.

        Args:
            model: The model to be watched.
            log_freq (int, optional): Logging frequency.
        """
        wandb.watch(model, log_freq=log_freq)

    def get_config(self):
        """
        Retrieves the configuration from W&B.

        Returns:
            dict: Configuration parameters from W&B.
        """
        return wandb.config
