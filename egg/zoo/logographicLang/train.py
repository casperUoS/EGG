
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import ToPILImage

import egg.core as core
from egg.core.reinforce_wrappers import PPOWrapper
from egg.zoo.logographicLang.archs import SketchEncoder, DiffDecoder, VisionEncoder
from egg.zoo.logographicLang.wrappers import AgentWrapper, Population, PopulationDiffGame
from egg.zoo.signal_game.archs import InformedSender, Receiver
from egg.zoo.signal_game_drawing.features import ImageNetFeat, ImagenetLoader, CIFAR10WithObj2ID
from egg.zoo.signal_game_drawing.archs import DrawSender, DrawReceiver, DrawReceiverClassifier, DrawSenderDiff
from egg.zoo.signal_game_drawing.wrappers import BezierReinforceWrapper, DiffRasterWrapper
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_root", default="", help="data root folder")
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
    parser.add_argument("--n_strokes",type=int,default=3,help="number of strokes")

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


#NOTE this excludes the edge pentalty loss, which
def loss_hinge(
    _sender_input, message, _receiver_input, receiver_output, labels, _aux_input
):
    hinge_loss = F.multi_margin_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return hinge_loss, {"acc": acc}

def get_game(config):
    if config['mode'] == "ds":

        sketch_decoder = DiffDecoder()
    else:
        sketch_decoder = None #Temp line

    sketch_encoder = SketchEncoder()
    vision_encoder = VisionEncoder(
        feat_size=config["feat_size"],
        hidden_size=config["sender_emb_size"],
        vision_path=opts.vgg_root,
    )

    agent = AgentWrapper(sketch_decoder,sketch_encoder,vision_encoder)
    population = Population()
    population.generate_population(agent,2)
    game = PopulationDiffGame(population,loss_hinge)

    return game

if __name__ == "__main__":
    wandb.login()

    opts = parse_arguments()

    config = dict(
        epochs=opts.n_epochs,
        classes=10,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        learning_rate=opts.lr,
        dataset='cifar10',
        game_size=opts.game_size,
        sender_entropy_coeff=0.0000001,
        receiver_entropy_coeff=0.1,
        all_classes=opts.all_classes,
        canvas_size=32,
        same_vgg_model=False,
        mode=opts.mode,
        diff_class=opts.diff_class if not None else False,
        sender_emb_size=128,  # originally 512, maybe go back to this
        feat_size=512,
        n_strokes=opts.n_strokes,
    )

    # data_folder = os.path.join(opts.root, "train/")
    cifar_path = "data/cifar10"
    dataset_exists = os.path.exists(os.path.join(cifar_path, "cifar-10-batches-py"))
    # dataset = ImageNetFeat(root=data_folder)

    project = "Diff Sketch New Framework"

    with wandb.init(project=project, config=config) as run:
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
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=callbacks,
            run=run,
            vgg_path=opts.vgg_root,
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