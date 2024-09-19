import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import v2
import os
import numpy as np


class F_ST_LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(context, input, slope):
        return F.leaky_relu(input, slope)

    @staticmethod
    def backward(context, grad_output):
        return grad_output, None


class ST_LeakyReLU(torch.nn.Module):
    def __init__(self, slope=0.01, *args, **kwargs):
        super(ST_LeakyReLU, self).__init__(*args, **kwargs)

        self.slope = slope

    def forward(self, input):
        return F_ST_LeakyReLU.apply(input, self.slope)


class ClassificationLoadedDataset(torch.utils.data.Dataset):
    """Classification Dataset in Memory"""

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert idx >= 0, "index must be positive"
        assert idx < len(self.x), "index must be within range"
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample


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
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Extracting a validation, rather than test, set
    # Last 10K samples taken as test
    if validation:
        # First shuffle the data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        nb_train_samples = len(x_train) - 10_000

        x_test = x_train[nb_train_samples:]
        y_test = y_train[nb_train_samples:]
        x_train = x_train[:nb_train_samples]
        y_train = y_train[:nb_train_samples]

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

    # Data to device (datasets small enough to fit directly)
    x_train = x_train.to(device).type(fltype)
    y_train = y_train.type(torch.LongTensor).to(device)

    x_test = x_test.to(device).type(fltype)
    y_test = y_test.type(torch.LongTensor).to(device)

    maxval = torch.max(x_train)
    x_train = x_train / maxval
    x_test = x_test / maxval

    if dataset_importer == "TIN":
        x_train = x_train.reshape(-1, 3, 64, 64)
        x_test = x_test.reshape(-1, 3, 64, 64)

        means = torch.mean(x_train, axis=(0, 2, 3))[None, :, None, None]
        stds = (torch.std(x_train, axis=(0, 2, 3)) + 1e-8)[None, :, None, None]
        x_train = (x_train - means) / stds
        x_test = (x_test - means) / stds

    return x_train, y_train, x_test, y_test


def construct_dataloaders(
    tv_dataset=torchvision.datasets.MNIST,
    batch_size=64,
    mean=None,
    std=None,
    validation=False,
    device="cpu",
):
    train_kwargs = {"batch_size": batch_size, "num_workers": 0, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "num_workers": 0, "shuffle": False}

    train_transforms, test_transforms = None, None
    if (
        tv_dataset == torchvision.datasets.CIFAR10
        or tv_dataset == torchvision.datasets.CIFAR100
        or tv_dataset == torchvision.datasets.MNIST
    ):

        if (
            tv_dataset == torchvision.datasets.CIFAR10
            or tv_dataset == torchvision.datasets.CIFAR100
        ):
            train_transforms = v2.Compose(
                [
                    v2.RandomCrop(32, padding=4),
                    v2.RandomHorizontalFlip(),
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            test_transforms = v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

        trainset = tv_dataset(
            root="./data", train=True, download=True, transform=train_transforms
        )
        if validation:
            trainset, testset = torch.utils.data.random_split(
                trainset, [len(trainset) - 10_000, 10_000]
            )
        else:
            testset = tv_dataset(
                root="./data", train=False, download=True, transform=test_transforms
            )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    else:
        # if tv_dataset == "TIN":
        train_transforms = v2.Compose(
            [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(56)]
        )
        test_transforms = v2.Compose([v2.CenterCrop(56)])

        x_train, y_train, x_test, y_test = load_dataset(
            tv_dataset, device, torch.float32, validation=validation, mean=mean, std=std
        )

        train_dataset = ClassificationLoadedDataset(x_train, y_train, train_transforms)
        test_dataset = ClassificationLoadedDataset(x_test, y_test, test_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


def test(
    model, device, train_test, test_loader, loud, loss_func, top5=False, num_classes=10
):
    model.eval()
    test_loss = 0
    correct = 0
    if top5:
        top5_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            onehots = torch.nn.functional.one_hot(target, num_classes).to(device)
            data, target = data.to(device), target.to(device)
            loss, output = model.test_step(data, target, onehots, loss_func)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, top5_pred = output.topk(5, dim=1, largest=True, sorted=True)
                top5_correct += (
                    top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
                )
        test_loss /= len(test_loader)

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

    if top5:
        return (
            test_loss,
            (
                (100.0 * correct / len(test_loader.dataset)),
                (100.0 * top5_correct / len(test_loader.dataset)),
            ),
        )
    return test_loss, (100.0 * correct / len(test_loader.dataset))


def update_metrics(
    model,
    metrics,
    device,
    train_test,
    loader,
    loss_func,
    epoch,
    loud=False,
    wandb=None,
    top5=False,
    num_classes=10,
):
    loss, acc = test(
        model, device, train_test, loader, loud, loss_func, top5, num_classes
    )
    metrics[train_test]["loss"].append(loss)
    metrics[train_test]["acc"].append(acc)

    if wandb is not None:
        wandb.log({f"{train_test}/loss": loss}, step=epoch)
        if top5:
            wandb.log(
                {f"{train_test}/top5": acc[1], f"{train_test}/acc": acc[0]}, step=epoch
            )
        else:
            wandb.log({f"{train_test}/acc": acc}, step=epoch)

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
    num_classes=10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        for o in optimizers:
            o.zero_grad()

        onehots = torch.nn.functional.one_hot(target, num_classes).to(device)
        data, target = data.to(device), target.to(device)
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


def init_metric(validation=False):
    if validation:
        return {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}
    else:
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
