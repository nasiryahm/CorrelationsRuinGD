import torch
import torchvision
import torch.nn.functional as F
import os
import numpy as np


class ClassificationLoadedDataset(torch.utils.data.Dataset):
    """Classification Dataset in Memory"""

    def __init__(self, x, y, y_onehot, transform=None):
        self.x = x
        self.y = y
        self.y_onehot = y_onehot
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert idx >= 0, "index must be positive"
        assert idx < len(self.x), "index must be within range"
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        y_onehot = self.y_onehot[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample, y_onehot


def format_tin_val(datadir):
    val_dir = datadir + "/tiny-imagenet-200/val"
    print("Formatting: %s" % val_dir)
    val_annotations = "%s/val_annotations.txt" % val_dir
    val_dict = {}
    with open(val_annotations, "r") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 6
            wnind = line[1]
            img_name = line[0]
            boxes = "\t".join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((img_name, boxes))
    assert len(val_dict) == 200
    for wnind, entries in val_dict.items():
        val_wnind_dir = "%s/%s" % (val_dir, wnind)
        val_images_dir = "%s/images" % val_dir
        val_wnind_images_dir = "%s/images" % val_wnind_dir
        os.mkdir(val_wnind_dir)
        os.mkdir(val_wnind_images_dir)
        wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
        f = open(wnind_boxes, "w")
        for img_name, box in entries:
            source = "%s/%s" % (val_images_dir, img_name)
            dst = "%s/%s" % (val_wnind_images_dir, img_name)
            os.system("cp %s %s" % (source, dst))
            f.write("%s\t%s\n" % (img_name, box))
        f.close()
    os.system("rm -rf %s" % val_images_dir)
    print("Cleaning up: %s" % val_images_dir)
    print("Formatting val done")


def load_dataset(dataset_importer, device, fltype, validation, mean, std):
    if dataset_importer == "TIN":
        if os.path.exists("./datasets/tiny-imagenet-200/y_train.npy"):
            print("Loading TinyImageNet")
            x_train = np.load("./datasets/tiny-imagenet-200/x_train.npy")
            y_train = np.load("./datasets/tiny-imagenet-200/y_train.npy").astype(int)
            x_test = np.load("./datasets/tiny-imagenet-200/x_test.npy")
            y_test = np.load("./datasets/tiny-imagenet-200/y_test.npy").astype(int)
        else:
            print("Down-Loading TinyImageNet")
            zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            torchvision.datasets.utils.download_and_extract_archive(
                url,
                "./datasets/",
                extract_root="./datasets/",
                remove_finished=True,
                md5=zip_md5,
            )

            format_tin_val("./datasets/")

            train_datasetpath = "./datasets/tiny-imagenet-200/train/"
            test_datasetpath = "./datasets/tiny-imagenet-200/val/"

            train_dataset = torchvision.datasets.ImageFolder(train_datasetpath)
            test_dataset = torchvision.datasets.ImageFolder(test_datasetpath)

            x_test = np.empty((len(test_dataset.targets), 3, 64, 64), dtype=np.float32)
            y_test = np.empty((len(test_dataset.targets)))
            for indx, (img, label) in enumerate(test_dataset.imgs):
                x_test[indx] = torchvision.transforms.ToTensor()(
                    test_dataset.loader(img).convert("RGB")
                )
                y_test[indx] = label
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)
            print("TinyImageNet test set loaded")

            np.save("./datasets/tiny-imagenet-200/x_test.npy", x_test)
            np.save("./datasets/tiny-imagenet-200/y_test.npy", y_test)

            x_train = np.empty(
                (len(train_dataset.targets), 3, 64, 64), dtype=np.float32
            )
            y_train = np.empty((len(train_dataset.targets)))
            for indx, (img, label) in enumerate(train_dataset.imgs):
                x_train[indx] = torchvision.transforms.ToTensor()(
                    train_dataset.loader(img).convert("RGB")
                )
                y_train[indx] = label
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            print("TinyImageNet training set loaded")

            np.save("./datasets/tiny-imagenet-200/x_train.npy", x_train)
            np.save("./datasets/tiny-imagenet-200/y_train.npy", y_train)

    else:
        train_dataset = dataset_importer("./datasets/", train=True, download=True)
        test_dataset = dataset_importer("./datasets/", train=False, download=True)

        # Loading dataset
        x_train = train_dataset.data
        y_train = train_dataset.targets
        x_test = test_dataset.data
        y_test = test_dataset.targets

    # Reshaping to flat digits
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Extracting a validation, rather than test, set
    # Last 10K samples taken as test
    if validation and not (dataset_importer == "tinyimagenet"):
        x_test = x_train[-10000:]
        y_test = y_train[-10000:]

        x_train = x_train[:50000]
        y_train = y_train[:50000]

    # # Picking only 1000 samples for training
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]

    # Squeezing out any excess dimension in the labels (true for CIFAR10/100)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test)
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test)

    # Creating onehot encoded targets
    y_train_onehot = torch.nn.functional.one_hot(y_train, torch.max(y_train) + 1)
    y_test_onehot = torch.nn.functional.one_hot(y_test, torch.max(y_train) + 1)

    # Normalizing data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if std is not None:
        d_mean = torch.mean(x_train, axis=0)
        d_std = torch.std(x_train, axis=0) + 1e-8
        x_train = ((x_train - d_mean) / d_std) * std + d_mean
        x_test = ((x_test - d_mean) / d_std) * std + d_mean
    if mean is not None:
        d_mean = torch.mean(x_train, axis=0)
        x_train = x_train - d_mean + mean
        x_test = x_test - d_mean + mean

    # Data to device (datasets small enough to fit directly)
    x_train = x_train.to(device).type(fltype)
    y_train = y_train.type(torch.LongTensor).to(device)
    y_train_onehot = y_train_onehot.to(device).type(fltype)

    x_test = x_test.to(device).type(fltype)
    y_test = y_test.type(torch.LongTensor).to(device)
    y_test_onehot = y_test_onehot.to(device).type(fltype)

    return x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot


def construct_dataloaders(
    tv_dataset=torchvision.datasets.MNIST,
    batch_size=64,
    mean=None,
    std=None,
    device="cpu",
):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}
    # if device[:5] == "cuda":
    # cuda_kwargs = {
    #     "num_workers": 1,
    #     "pin_memory": True,
    #     "shuffle": False,
    # }  # Shuffle to false for replicability
    # train_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)

    x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot = load_dataset(
        tv_dataset, device, torch.float32, validation=False, mean=mean, std=std
    )

    train_dataset = ClassificationLoadedDataset(x_train, y_train, y_train_onehot)
    test_dataset = ClassificationLoadedDataset(x_test, y_test, y_test_onehot)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


def test(model, device, train_test, test_loader, loud, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, onehots in test_loader:
            loss, output = model.test_step(data, target, onehots, loss_func)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if loud:
        print(
            "\n{} set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                train_test,
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss, (100.0 * correct / len(test_loader.dataset))


def update_metrics(
    model, metrics, device, train_test, loader, loss_func, epoch, loud=False, wandb=None
):
    loss, acc = test(model, device, train_test, loader, loud, loss_func)
    metrics[train_test]["loss"].append(loss)
    metrics[train_test]["acc"].append(acc)

    if wandb is not None:
        wandb.log({f"{train_test}/loss": loss, f"{train_test}/acc": acc}, step=epoch)

    return metrics


def train(
    model,
    device,
    train_loader,
    optimizers,
    epoch,
    loss_func,
    log_interval=100,
    loud=False,
):
    model.train()
    for batch_idx, (data, target, onehots) in enumerate(train_loader):
        for o in optimizers:
            o.zero_grad()

        loss = model.train_step(data, target, onehots, loss_func)

        for o in optimizers:
            o.step()
        if (batch_idx % log_interval == 0) and loud:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def init_metric():
    return {"train": {"loss": [], "acc": []}, "test": {"loss": [], "acc": []}}


def plot_metrics(metrics):
    plt.subplot(2, 1, 1)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(metrics["loss"]["train"])
    plt.plot(metrics["loss"]["test"])
    plt.subplot(2, 1, 2)
    plt.ylabel("accuracy")
    plt.xlabel("Epoch")
    plt.plot(metrics["acc"]["train"])
    plt.plot(metrics["acc"]["test"])
