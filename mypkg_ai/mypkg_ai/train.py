import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PyQt5.QtCore import QLibraryInfo
import pandas as pd
from termcolor import colored

import os
import time
from collections import defaultdict

from mypkg_ai.dataset import Dataset
from mypkg_ai.network import Model
import mypkg_ai.config as config
from mypkg_ai.utils import time_it


@time_it
def initialize_dataset(
        train_dataset_path: str = config.TRAIN_DATASET_PATH,
        val_dataset_path: str = config.VAL_DATASET_PATH,
        test_dataset_path: str = config.TEST_DATASET_PATH,
    ) -> tuple[
        Dataset,
        Dataset,
        Dataset,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]]
    ]:
    """
    Initialize the training, validation and testing datasets.

    Args:
        train_dataset_path (str): The path to the training dataset. Defaults to config.TRAIN_DATASET_PATH.
        val_dataset_path (str): The path to the validation dataset. Defaults to config.VAL_DATASET_PATH.
        test_dataset_path (str): The path to the testing dataset. Defaults to config.TEST_DATASET_PATH.

    Returns:
        Dataset: The training dataset.
        Dataset: The validation dataset.
        Dataset: The testing dataset.
        list[tuple[torch.Tensor, torch.Tensor]]: The training dataset.
        list[tuple[torch.Tensor, torch.Tensor]]: The validation dataset.
        list[tuple[torch.Tensor, torch.Tensor]]: The testing dataset.
    """
    train_dataset = Dataset.load(train_dataset_path)
    train_data, train_signs = train_dataset.inputs, train_dataset.targets
    train_data = list(zip(train_data, train_signs))
    
    val_dataset = Dataset.load(val_dataset_path)
    val_data, val_signs = val_dataset.inputs, val_dataset.targets
    val_data = list(zip(val_data, val_signs))

    test_dataset = Dataset.load(test_dataset_path)
    test_data, test_signs = test_dataset.inputs, test_dataset.targets
    test_data = list(zip(test_data, test_signs))

    print(colored(f"**** {len(train_data)} training, {len(val_data)} validation and "
            f"{len(test_data)} test samples", "light_green"))

    return train_dataset, val_dataset, test_dataset, train_data, val_data, test_data


@time_it
def create_data_loaders(
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the training, validation and testing data loaders.
    
    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.

    Returns:
        DataLoader: The training data loader.
        DataLoader: The validation data loader.
        DataLoader: The testing data loader.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NB_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NB_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NB_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader


@time_it
def save_data_as_csv(
        test_data: list[tuple[torch.Tensor, torch.Tensor]],
        val_data: list[tuple[torch.Tensor, torch.Tensor]],
        train_data: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
    """
    Save the training, validation and testing split data as CSV.

    Args:
        test_data (list[tuple[torch.Tensor, torch.Tensor]]): The testing data.
        val_data (list[tuple[torch.Tensor, torch.Tensor]]): The validation data.
        train_data (list[tuple[torch.Tensor, torch.Tensor]]): The training data.
    """
    print(colored("**** saving training, validation and testing split data as CSV...", "light_green"))
    time_counter = time.perf_counter()
    with open(config.TEST_PATH, "w") as f:
        f.write("\n".join([','.join(map(str, row)) for row in test_data]))
    with open(config.VAL_PATH, "w") as f:
        f.write("\n".join([','.join(map(str, row)) for row in val_data]))
    with open(config.TRAIN_PATH, "w") as f:
        f.write("\n".join([','.join(map(str, row)) for row in train_data]))
    print(colored(f"**** time to save CSV files: {time.perf_counter() - time_counter:.4f}s", "light_cyan"))


@time_it
def initialize_network(load_model: bool = config.LOAD_MODE) -> Model:
    """
    Initialize the network by loading the last model or creating a new one.

    Args:
        load_model (bool): Whether to load the last model. Defaults to config.LOAD_MODE.

    Returns:
        Model: The initialized network.
    """
    print(colored("**** initializing network...", "light_green"))
    if load_model:
        model = torch.load(config.LAST_MODEL_PATH).to(config.DEVICE)
    else:
        model = Model().to(config.DEVICE)
    return model


@time_it
def load_plot_data() -> defaultdict[list]:
    """
    Load the plot data from the CSV file.

    Returns:
        defaultdict[list]: The loaded plot data.
    """
    df = pd.read_csv(config.PLOT_DATA_PATH)
    plots = defaultdict(list)
    plots['Training loss'] = df['Training loss'].to_list()
    plots['Validation loss'] = df['Validation loss'].to_list()
    plots['Training accuracy'] = df['Training accuracy'].to_list()
    plots['Validation accuracy'] = df['Validation accuracy'].to_list()
    return plots


def compute_loss(
        model: Model,
        optimizer: Adam,
        loader: DataLoader,
        back_prop: bool = False,
        progress_desc: str = "Progress",
        display_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Function to compute loss over a batch.

    Args:
        model (Model): The model.
        optimizer (Adam): The optimizer.
        loader (DataLoader): The data loader.
        back_prop (bool): Whether to back propagate. Defaults to False.
        progress_desc (str): The progress description. Defaults to "Progress".
        display_loss (bool): Whether to display the loss. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The total loss and the accuracy.
    """
    # Initialize the total loss and number of correct predictions
    total_loss, correct = 0, 0
    len_dataset = len(loader.dataset)

    # Loop over batches of the training set
    with tqdm(
            total=len(loader),
            desc=f"{progress_desc}: ",
            unit='batch',
            ascii=".=",  # "=" for filled, "." for not filled
        ) as pbar:
        for i, batch in enumerate(loader):
            # Send the inputs and training annotations to the device
            inputs_batch, targets = batch
            inputs_batch = inputs_batch.to(config.DEVICE)
            targets = targets.to(config.DEVICE).float()

            # Perform a forward pass and calculate the training loss
            predict = model(inputs_batch)
            # NOTE: Use categorical cross-entropy loss as an example
            batch_loss = torch.nn.functional.cross_entropy(predict, targets, reduction="sum")

            # Zero out the gradients, perform backprop & update the weights
            if back_prop:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # Add the loss to the total training loss so far
            total_loss += batch_loss
            correct += (predict.argmax(1) == targets).type(torch.float).sum().item()

            # Update the progress bar
            if display_loss:
                pbar.set_postfix({'loss': batch_loss.item()})
            pbar.update(1)

    # Return sample-level averages of the loss and accuracy
    return total_loss / len_dataset, correct / len_dataset


@time_it
def make_plots(plots: dict) -> None:
    """
    Make the training plot.

    Args:
        plots (dict): The plot data.
    """
    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(plots['Training loss'], label="Train Loss")
    plt.plot(plots['Validation loss'], label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    plt.subplot(2, 1, 2)
    plt.plot(plots['Training accuracy'], label="Train Acc")
    plt.plot(plots['Validation accuracy'], label="Val Acc")
    plt.title("Training/Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # Save the training plot
    plt.tight_layout()
    plt.savefig(config.PLOT_PATH)
    # plt.show() # Uncomment if you want to display the plot as well


@time_it
def save_plot_data(plots: dict) -> None:
    """
    Save the plot data.

    Args:
        plots (dict): The plot data.
    """
    # Save the plot data
    df = pd.DataFrame.from_dict(plots)
    df.to_csv(config.PLOT_DATA_PATH, index=False)


@time_it
def train_model():
    """Train the model."""

    # Set the Qt plugin path to the path of the Qt plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # Initialize the dataset
    train_dataset, val_dataset, test_dataset, train_data, val_data, test_data = initialize_dataset()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Save training, validation and testing split data as CSV
    save_data_as_csv(test_data, val_data, train_data)

    # Create our model and upload to the current device
    model = initialize_network()

    # Initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(model.parameters(), lr=config.INIT_LR)
    print(model)

    # Initialize history variables for future plot
    if config.LOAD_MODE:
        plots = load_plot_data()
    else:
        plots = defaultdict(list)

    # loop over epochs
    print(colored("**** training the network...", "light_green"))
    best_val_acc = 0.0
    best_val_loss = float('inf')
    start_time = time.perf_counter()
    for e in range(config.NUM_EPOCHS_DONE, config.NUM_EPOCHS):
        # Print epoch number
        print(colored(f"**** EPOCH: {e + 1}/{config.NUM_EPOCHS}", "light_magenta"))

        # Set model in training mode & backpropagate train loss for all batches
        model.train()

        # Do not use the returned loss
        # The loss of each batch is computed with a "different network"
        # as the weights are updated per batch
        compute_loss(train_loader, back_prop=True, progress_desc="Training", display_loss=True)

        # Switch off autograd
        with torch.no_grad():
            # Set the model in evaluation mode and compute validation loss
            model.eval()
            train_loss, train_acc, train_acc_signs = compute_loss(
                train_loader,
                progress_desc="Validation on train"
            )
            val_loss, val_acc, val_acc_signs = compute_loss(
                val_loader,
                progress_desc="Validation on val",
            )

        # Update our training history
        plots['Training loss'].append(train_loss.cpu())
        plots['Training accuracy'].append(train_acc)

        plots['Validation loss'].append(val_loss.cpu())
        plots['Validation accuracy'].append(val_acc)

        # Plot the training loss and accuracy
        save_plot_data(plots)

        # print the model training and validation information
        print(colored(f"Train loss: {train_loss:.8f}, Train accuracy: {train_acc:.8f}", "light_cyan"))
        print(colored(f"Val loss: {val_loss:.8f}, Val accuracy: {val_acc:.8f}", "light_cyan"))
        print(colored(f"Train accuracy signs: {train_acc_signs}", "light_cyan"))
        print(colored(f"Val accuracy signs: {val_acc_signs}", "light_cyan"))

        # Store model with highest accuracy, lowest loss
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss

            # Serialize the model to disk
            print(colored("**** saving BEST object detector model...", "light_green"))
            # When a network has dropout and / or batchnorm layers
            # one needs to explicitly set the eval mode before saving
            model.eval()
            torch.save(model, config.BEST_MODEL_PATH)

        print(colored("**** saving LAST object detector model...", "light_green"))
        model.eval()
        # torch.save(model, f"{config.mypkg_ai_LAST_MODEL_PATH[:-4]}_epoch_{e}.pth")
        torch.save(model, config.LAST_MODEL_PATH)

    end_time = time.perf_counter()
    print(colored(f"**** total time to train the model: {end_time - start_time:.2f}s", "light_blue"))

    # plot the training loss and accuracy
    save_plot_data(plots)
    make_plots(plots)
