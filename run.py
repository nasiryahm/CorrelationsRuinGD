from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
import os
import numpy as np
from utils import *
from models import DecorNet, DecorConvNet, PerturbNet
from bp import BPLinear, BPConv2d
from decor import DecorLinear
from fa import FALinear, FAConv2d
from np import NPLinear
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig


def train_network(
    batch_size=256,
    dataset="MNIST",
    device="cuda",
    bias=True,
    regularizer_strength=0.0,
    decorrelation_method="copi",
    decor_lr=1e-3,
    n_hidden_layers=3,
    hidden_layer_size=1000,
    layer_type=BPLinear,
    loss_func_type="CCE",  # "MSE"
    optimizer_type="Adam",
    fwd_lr=1e-2,
    seed=42,
    nb_epochs=10,
    loud=True,
    layer_kwargs={},
    decor_layer_kwargs={},
    wandb=None,
):

    betas = [0.9, 0.9999]
    eps = 1e-8

    # Initializing random seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    tv_dataset = dataset
    if dataset == "MNIST":
        tv_dataset = torchvision.datasets.MNIST
    elif dataset == "CIFAR10":
        tv_dataset = torchvision.datasets.CIFAR10
    elif dataset == "CIFAR100":
        tv_dataset = torchvision.datasets.CIFAR100

    train_loader, test_loader = construct_dataloaders(
        tv_dataset, batch_size=batch_size, device=device
    )

    # If dataset is CIFAR, change input shape
    in_size = 28 * 28
    out_size = 10
    if tv_dataset == torchvision.datasets.CIFAR10:
        in_size = 32 * 32 * 3
    if tv_dataset == torchvision.datasets.CIFAR100:
        in_size = 32 * 32 * 3
        out_size = 100
    if tv_dataset == "TIN":
        in_size = 64 * 64 * 3
        out_size = 200

    # Initialize model
    if layer_type in [BPLinear, FALinear]:
        model_type = DecorNet
    if layer_type in [NPLinear]:
        model_type = PerturbNet
        distribution = torch.distributions.Normal(
            torch.tensor([0.0]).to(torch.float32).to(device),
            torch.tensor([1.0]).to(device),
        )
        dist_sampler = lambda x: distribution.sample([1] + x).squeeze_(-1).sum(0)
        layer_kwargs = {
            "sigma": 1e-6,
            "dist_sampler": dist_sampler,
        }
    if layer_type in [BPConv2d, FAConv2d]:
        model_type = DecorConvNet
        in_size = [3, 64, 64]
        if dataset in ["CIFAR10", "CIFAR100"]:
            in_size = [3, 32, 32]
        if dataset == "MNIST":
            in_size = [1, 28, 28]

    model = model_type(
        in_size=in_size,
        out_size=out_size,
        n_hidden_layers=n_hidden_layers,
        hidden_size=hidden_layer_size,
        layer_type=layer_type,
        decorrelation_method=decorrelation_method,
        biases=bias,
        layer_kwargs=layer_kwargs,
        decor_layer_kwargs=decor_layer_kwargs,
    )
    model.to(device)

    # Initialize metric storage
    metrics = init_metric()

    # Define optimizers
    fwd_optimizer = None
    if optimizer_type == "Adam":
        fwd_optimizer = torch.optim.Adam(
            model.get_fwd_params(),
            betas=betas,
            eps=eps,
            lr=fwd_lr,
        )
    elif optimizer_type == "SGD":
        fwd_optimizer = torch.optim.SGD(
            model.get_fwd_params(), lr=fwd_lr, weight_decay=regularizer_strength
        )

    optimizers = [fwd_optimizer]
    if decorrelation_method is not None:
        decor_optimizer = torch.optim.SGD(model.get_decor_params(), lr=decor_lr)
        optimizers.append(decor_optimizer)

    loss_func = None
    if loss_func_type == "CCE":
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif loss_func_type == "MSE":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = lambda input, target, onehot: torch.sum(
            loss_obj(input, onehot), axis=1
        )

    # Train loop
    for e in tqdm(range(nb_epochs + 1), disable=not loud):
        metrics = update_metrics(
            model,
            metrics,
            device,
            "train",
            train_loader,
            loss_func,
            e,
            loud=loud,
            wandb=wandb,
        )
        metrics = update_metrics(
            model,
            metrics,
            device,
            "test",
            test_loader,
            loss_func,
            e,
            loud=loud,
            wandb=wandb,
        )
        if e < nb_epochs:
            train(model, device, train_loader, optimizers, e, loss_func, loud=False)
        if np.isnan(metrics["test"]["loss"][-1]) or np.isnan(
            metrics["train"]["loss"][-1]
        ):
            print("NaN detected, aborting training")
            break
    return metrics


@hydra.main(version_base="1.3", config_path="conf/", config_name="config")
def run(config: DictConfig) -> None:
    torch.set_num_threads(2)

    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.init(
        config=cfg,
        entity=config.wandb.entity,
        project=config.wandb.project,
        mode=config.wandb.mode,
    )

    if config.decorrelation_method == "None":
        config.decorrelation_method = None

    # For now foldiak is too slow unfortunately
    if config.decorrelation_method == "foldiak":
        exit()

    layer_type = BPLinear
    assert config.layer_type in [
        "BP",
        "FA",
        "NP",
        "BPConv",
        "FAConv",
    ], "Invalid layer type"
    if config.layer_type == "FA":
        layer_type = FALinear
    elif config.layer_type == "NP":
        layer_type = NPLinear
    elif config.layer_type == "BPConv":
        layer_type = BPConv2d
    elif config.layer_type == "FAConv":
        layer_type = FAConv2d

    metrics = train_network(
        batch_size=config.batch_size,
        dataset=config.dataset,
        device=config.device,
        bias=config.bias,
        regularizer_strength=config.regularizer_strength,
        decorrelation_method=config.decorrelation_method,
        decor_lr=config.decor_lr,
        n_hidden_layers=config.n_hidden_layers,
        hidden_layer_size=config.hidden_layer_size,
        layer_type=layer_type,
        loss_func_type=config.loss_func_type,
        optimizer_type=config.optimizer_type,
        fwd_lr=config.fwd_lr,
        seed=config.seed,
        nb_epochs=config.nb_epochs,
        loud=config.loud,
        wandb=wandb,
    )

    print(metrics)


if __name__ == "__main__":
    run()
