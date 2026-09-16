
import argparse
import os

import torch
import torch.nn.functional as F
from model_utilities.datasets import make_wids_sampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.v2 import ToPILImage

import wandb
from egg import core
from egg.zoo.logographicLang.archs import DiffDecoder, SketchEncoder, VisionEncoder
from egg.zoo.logographicLang.features import (
    CIFAR10WithObj2ID,
    ImageNet50WIDSFeat,
    ImagenetLoader,
)
from egg.zoo.logographicLang.wrappers import (
    AgentWrapper,
    Population,
    PopulationDiffGame,
)
from model_utilities.datasets.imagenet_subsets_wids import ImageNet50WIDS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_root", default="", help="data root folder")
    # 2-agents specific parameters
    parser.add_argument(
        "--tau_s", type=float, default=10.0, help="Sender Gibbs temperature"
    )
    parser.add_argument(
        "--game_size", type=int, default=2, help="Number of images seen by an agent"
    )
    parser.add_argument("--same", type=int, default=0, help="Use same concepts")
    parser.add_argument("--embedding_size", type=int, default=50, help="embedding size")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="hidden size (number of filters informed sender)",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="Batches in a single training/validation epoch",
    )
    parser.add_argument("--inf_rec", type=int, default=0, help="Use informed receiver")
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.",
    )
    parser.add_argument("--gs_tau", type=float, default=1.0, help="GS temperature")
    parser.add_argument("--sample_mode", default="all", help="'all': display all classes. 'single' display one class, 'double' display two classes")
    parser.add_argument("--all_classes", action=argparse.BooleanOptionalAction, help="Turns signal game into classification game")
    parser.add_argument("--diff_class", action=argparse.BooleanOptionalAction, help="wether to get different instance of class for receiver")
    parser.add_argument("--n_strokes",type=int, default=3, help="number of strokes")
    parser.add_argument("--pop_size", type=int, default=2, help="size of poputlation")
    parser.add_argument("--dataset", default="imageNet10", help="choose dataset")

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


#NOTE this excludes the edge pentalty loss, which
def loss_hinge(
     receiver_output, labels
):
    hinge_loss = F.multi_margin_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return hinge_loss, {"acc": acc}

def get_game(config):
    if config['mode'] == "ds":

        sketch_decoder = DiffDecoder()
    else:
        sketch_decoder = None #Temp line

    sketch_encoder = SketchEncoder(embedding_size=config["z_dim"])
    vision_encoder = VisionEncoder(
        feat_size=config["feat_size"],
        hidden_size=config["sender_emb_size"],
        vision_path=opts.vision_root,
        z_dim=config["z_dim"]
    )

    agent = AgentWrapper(sketch_encoder,vision_encoder, sketch_decoder, config)
    population = Population()
    population.generate_population(agent,config["pop_size"])
    game = PopulationDiffGame(population,loss_hinge)

    return game

if __name__ == "__main__":
    wandb.login()

    project = "SKEGG"

    opts = parse_arguments()
    device = opts.device
    print("Device =", device, flush=True)

    config = {
        "epochs": opts.n_epochs,
        "classes": 10,
        "batch_size": opts.batch_size,
        "batches_per_epoch": opts.batches_per_epoch,
        "learning_rate": opts.lr,
        "game_size": opts.game_size,
        "sender_entropy_coeff": 0.0000001,
        "receiver_entropy_coeff": 0.1,
        "all_classes": opts.all_classes,
        "canvas_size": 32,
        "same_vision_model": False,
        "mode": opts.mode,
        "diff_class": opts.diff_class if not None else False,
        "sender_emb_size": 512,  # originally 512, maybe go back to this
        "z_dim": 20,
        "feat_size": 2048,
        "n_strokes": opts.n_strokes,
        "pop_size": opts.pop_size,
        "dataset": opts.dataset
    }

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if config["dataset"] == "imageNet10":
        train_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/train", transform=transform_train)
        test_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/val", transform=transform_test)
        train_sampler = make_wids_sampler(train_dataset)
        test_sampler = make_wids_sampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],sampler=train_sampler, num_workers=5)
        validation_loader = DataLoader(test_dataset, batch_size=config['batch_size'],sampler=test_sampler, num_workers=5)

    else:
        cifar_path = "data/cifar10"
        dataset_exists = os.path.exists(os.path.join(cifar_path, "cifar-10-batches-py"))
        # dataset = ImageNetFeat(root=data_folder)
        
        train_dataset = CIFAR10WithObj2ID(cifar_path, train=True, download=not dataset_exists)
        test_dataset = CIFAR10WithObj2ID(cifar_path, train=False, download=not dataset_exists)
        train_loader = ImagenetLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            opt=opts,
            batches_per_epoch=config['batches_per_epoch'],
            seed=None,
            diff_class=config['diff_class'],
        )
        validation_loader = ImagenetLoader(
            test_dataset,
            opt=opts,
            batch_size=config['batch_size'],
            batches_per_epoch=config['batches_per_epoch'],
            seed=21,
            diff_class=config['diff_class'],
        )
    
    game = get_game(config)
    optimizer = core.build_optimizer(game.parameters())
    callback = None
    if opts.mode == "gs":
        callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))

    with wandb.init(project=project, config=config) as run:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=callbacks,
            run=run,
            vision_path=opts.vision_root,
            device=device,
            game_size=config['game_size']
        )

        trainer.train(n_epochs=config['epochs'])

        print("Generating sample sketch...")
        val_loss, interaction = trainer.eval()

        # symbolicity_loss, symbolicity_acc, semantic_cor = trainer.symbolicity_eval(epochs=10)
        #
        # print("Symbolicity score:", symbolicity_loss)
        # print("Symbolicity accuracy:", symbolicity_acc)
        # print("Semanticity score:", semantic_cor)
        #
        # wandb.log({"symbolicity_loss": symbolicity_loss})
        # wandb.log({"symbolicity_acc": symbolicity_acc})
        # wandb.log({"semantic_cor": semantic_cor})

        for sample_mode in ["all","single","double"]:
        
            sketches = interaction.message.detach().cpu()
            splines = interaction.sender_output.detach().cpu()
            sender_input = interaction.sender_input.detach().cpu()
            # receiver_input = interaction.receiver_input.detach().cpu()
            receiver_output = interaction.receiver_output.detach().cpu()
            labels = interaction.labels.detach().cpu()
            # edge_penalty = interaction.edge_penalty.detach().cpu() if not None else 0.0

            # 3. Plot and save one sample
            import matplotlib.pyplot as plt

            # Pick the first image in the batch
            sample = sketches[0]

            # Remove the channel dimension if it exists (e.g., convert 1x28x28 to 28x28)
            if sample.ndim == 3:
                sample = sample.squeeze(0)

            # print("sample =", splines[0])
            # print(splines.shape)
            #
            # print("sender_input =", sender_input[0][0])
            # print("sender_shape=", sender_input.shape )
            # print("reciever_input =", reciever_input[0][0])
            # print("reciever_shape=", reciever_input.shape )

            # print("reciever_output=",receiver_output)
            # print("labels=",labels)

            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

            num_samples = min(32, sketches.size(0))
            max_rows = 8
            num_cols = ((num_samples - 1) // max_rows) + 1  # Calculate number of column pairs needed
            num_rows = min(num_samples, max_rows)

            # Create a grid: rows x (2 * num_cols) since each sample needs 2 subplots
            fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(8 * num_cols, num_rows * 3))

            # Handle single column case
            if num_cols == 1 and num_rows == 1:
                axes = axes.reshape(1, -1)
            elif num_cols == 1:
                axes = axes.reshape(num_rows, -1)
            elif num_rows == 1:
                axes = axes.reshape(-1, 2 * num_cols)



            # print("single_class_idx", len(single_class_idx))
            # print("single_class_idx", single_class_idx)

            if sample_mode == "single":
                single_class_idx = (labels == 0).nonzero(as_tuple=True)[0]
                sketches = sketches[single_class_idx]
                labels = labels[single_class_idx]
                # edge_penalty = edge_penalty[single_class_idx]
                # edge_penalty = 0.0
                if sender_input.ndim == 5:
                    sender_input = sender_input.index_select(dim=1, index=single_class_idx)
                else:
                    sender_input = sender_input[single_class_idx]

            if sample_mode == "double":
                class1_idx = (labels == 1).nonzero(as_tuple=True)[0][0:(num_samples//2)]
                class2_idx = (labels == 6).nonzero(as_tuple=True)[0][0:(num_samples//2)]
                sketches = torch.cat((sketches[class1_idx], sketches[class2_idx]))
                labels = torch.cat([labels[class1_idx], labels[class2_idx]])
                # edge_penalty = torch.cat([edge_penalty[class1_idx], edge_penalty[class2_idx]])
                # edge_penalty = 0.0
                if sender_input.ndim == 5:
                    sender_input = torch.cat([sender_input.index_select(dim=1, index=class1_idx), sender_input.index_select(dim=1, index=class2_idx)], dim=1)
                else:
                    sender_input = torch.cat([sender_input[class1_idx], sender_input[class2_idx]])

            long_img = []

            for i in range(num_samples):
                # Calculate which column pair and row this sample belongs to
                col_pair = i // max_rows
                row = i % max_rows

                sketch_sample = sketches[i]

                if sender_input.ndim == 5:
                    original_sample = sender_input[0, i]
                else:
                    original_sample = sender_input[i]

                if sketch_sample.ndim == 3:
                    sketch_sample = sketch_sample.squeeze(0)

                if original_sample.ndim == 3:
                    original_sample = original_sample.permute(1, 2, 0)

                if original_sample.max() > 1.0:
                    original_sample = original_sample / 255.0

                long_img.append(original_sample)
                long_img.append(sketch_sample.unsqueeze(2).expand(-1, -1, 3))

                class_idx = labels[i].item()
                class_name = class_idx
                # edge_penalty_sample = edge_penalty[i]
                # edge_penalty_sample = 0.0

                # Calculate column indices for original and sketch
                orig_col = col_pair * 2
                sketch_col = col_pair * 2 + 1

                # Plot original image
                axes[row, orig_col].imshow(original_sample)
                axes[row, orig_col].set_title(f"Original: {class_name}")
                axes[row, orig_col].axis('off')

                # Plot sketch
                axes[row, sketch_col].imshow(sketch_sample, cmap='gray', origin='lower')
                axes[row, sketch_col].set_title(f"Sketch: {class_name}")
                axes[row, sketch_col].axis('off')


            # Hide unused subplots
            for i in range(num_samples, num_rows * num_cols):
                col_pair = i // max_rows
                row = i % max_rows
                axes[row, col_pair * 2].axis('off')
                axes[row, col_pair * 2 + 1].axis('off')

            # long_img = torch.cat(long_img, dim=1).detach().cpu()

            plt.tight_layout()
            plt.suptitle("Original Images vs Sketches from Trained Sender", y=1.00)
            wandb.log({f"plot_{sample_mode}": fig})

            # long_img = ToPILImage()(long_img.permute(2, 0, 1))
            # wandb.log({f"long_img_{sample_mode}": wandb.Image(long_img)})