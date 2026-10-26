import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import InterpolationMode, rotate

from egg.core import LoggingStrategy


class SenderWrapper(nn.Module):
    def __init__(self, vision_encoder, sketch_decoder, hidden_size, z_dim, samples=1):
        super().__init__()
        self. vision_encoder = vision_encoder
        self.sketch_decoder = sketch_decoder
        self.samples = samples

        self.fc_mu = nn.Linear(hidden_size, z_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_size, z_dim, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + std*eps, std

    def forward(self, x):
        x, vision_aux = self.vision_encoder(x)
        logvar = self.fc_logvar(x)
        mu = self.fc_mu(x)
        z, _std = self.reparameterize(mu, logvar)
        if self.samples > 1:
            z = z.repeat(self.samples) #portentially do torch.interleave
        output, sketch_enc_aux = self.sketch_decoder(z)
        output = output.unsqueeze(1)
        sender_aux = (sketch_enc_aux | vision_aux) 
            
        sender_aux['mu'] = mu
        sender_aux['log_var'] = logvar

        return output, sender_aux


class ReceiverWrapper(nn.Module):
    def __init__(self, vision_encoder, sketch_encoder, config, hidden_size=128):
        super().__init__()
        self.sketch_encoder = sketch_encoder
        self.vision_encoder = vision_encoder
        self.game_size = config["game_size"]
        self.isOptimized = config["isOptimized"]

        self.mlp = nn.Sequential(
            nn.Linear(config["emb_size"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config["reciever_z_dim"]))

        # self.vis_rep = nn.Sequential(nn.SELU(), nn.Linear(config["sender_emb_size"], config["sketch_emb_size"]))

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(logvar)
    #     return mu + std*eps, std

    def return_embeddings(self, x):
        # embed each image (left or right)
        x = x.transpose(0,1)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h, _ = self.vision_encoder(h)
            h = self.mlp(h)
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            # h_i are batch_size x embedding_size
            h_i = h.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h

    def return_embeddings_optimized(self,x):
        x = x.squeeze(2)
        b_1, b_2 = x.shape[:2]
        batch_size = b_1 * b_2 
        # print("batch size", batch_size, flush=True)
        # print("game_size", self.game_size, flush=True)
        x_flat = x.flatten(start_dim=0, end_dim=1)
        h, _= self.vision_encoder(x_flat)
        h = self.mlp(h)
        h = h.reshape(b_1, b_2, -1)

        h = torch.roll(h, -1,0)
        h_all = []
        for i in range(self.game_size):
            h_i = torch.roll(h, -i,1) #x,y,emb
            h_i = torch.flatten(h_i, start_dim=0, end_dim=1).squeeze()
            h_all.append(h_i)

        out = torch.stack(h_all, dim=1) #shape: ((50*4), 5, E)
        shuffle = torch.stack(
            [torch.randperm(self.game_size, device=out.device) for _ in range(batch_size)]
        )
        # print("shuffle", shuffle.shape, flush=True)
        out = out.gather(1, shuffle.unsqueeze(-1).repeat(1,1,out.shape[-1]))

        labels = shuffle.argmin(dim=1)
        return out, labels

        #to make it fit nicely, we would want batch_size x game_size x embedding_size 

    def forward(self, signal, x):
        auxdata = {}
        if self.isOptimized:
            emb, labels = self.return_embeddings_optimized(x)
            auxdata["labels"] = labels
        else:
            emb = self.return_embeddings(x)

        h_s, _sk_enc_aux = self.sketch_encoder(signal)
        # embd_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        # print("emb size:", emb.shape, flush=True)
        # print("h_s size:", h_s.shape, flush=True)
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size

        log_probs = F.log_softmax(out, dim=1)
        auxdata["receiver_features"] = h_s

        return log_probs, auxdata



class AgentWrapper(nn.Module):
    def __init__(self, sketch_encoder, vision_encoder, sketch_decoder, config):
        super().__init__()
        self.sender = SenderWrapper(
            vision_encoder,
            sketch_decoder,
            config["emb_size"],
            config["sender_z_dim"]
        )
        self.receiver = ReceiverWrapper(vision_encoder, sketch_encoder, config)

    def get_sender(self):
        return self.sender

    def get_receiver(self):
        return self.receiver

    def forward(self,*x,mode):
        if mode == "s":
            return self.sender(*x)
        if mode == "r":
            return self.receiver(*x)
        return None


class Population(nn.Module):
    def __init__(self):
        super().__init__()
        self.population = nn.ModuleList([])

    def generate_population(self, agent, size):
        for _ in range(size):
            agent_copy = deepcopy(agent)  # Create a new instance
            self.add_agent(agent_copy)

    def add_agent(self, agent):
        self.population.append(agent)

    def __len__(self):
        return len(self.population)

    def get_pair(self):
        if len(self.population) < 2:
            raise IndexError('Population must have at least two agents')

        agent_index = random.sample(range(len(self.population)), 2)
        sender = self.population[agent_index[0]]
        receiver = self.population[agent_index[1]]

        return sender, receiver



class PopulationDiffGame(nn.Module):

    def __init__(
            self,
            population,
            loss,
            config,
            train_logging_strategy = None,
            test_logging_strategy = None,
            max_rotation = 30.0,
    ):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.
        """
        super().__init__()
        self.population = population
        self.loss = loss
        self.max_rotation = max_rotation
        self.train_logging_strategy = (
            LoggingStrategy(
                store_sender_input=False,
                store_receiver_input=False,
                store_message=False,
                store_receiver_output=False,
                store_sender_output=False,
                store_vgg_features=False,
                store_receiver_features=False,
            )
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy( #TODO this needs to be changed so that we can actually anylise results, this would involve putting the results into a fixed torch.zeros_lik(50 ...)
                store_sender_input=False,
                store_receiver_input=False,
                store_message=False,
                store_receiver_output=False,
                store_sender_output=False,
                store_vgg_features=False,
                store_receiver_features=False,
            )
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.isOptimized = config["isOptimized"]

    # def rotate_message(self, message):
    #     batch_size = message.size(0)
    #     angles = torch.empty(batch_size, device=message.device).uniform_(
    #         -self.max_rotation, self.max_rotation
    #     )
    #     angles = angles * torch.pi / 180.0

    #     cos_angles = torch.cos(angles)
    #     sin_angles = torch.sin(angles)
    #     theta = torch.zeros(
    #         batch_size, 2, 3, device=message.device, dtype=message.dtype
    #     )
    #     theta[:, 0, 0] = cos_angles
    #     theta[:, 0, 1] = -sin_angles
    #     theta[:, 1, 0] = sin_angles
    #     theta[:, 1, 1] = cos_angles

    #     grid = F.affine_grid(theta, message.size(), align_corners=False)
    #     return F.grid_sample(
    #         message,
    #         grid,
    #         mode="bilinear",
    #         padding_mode="zeros",
    #         align_corners=False,
    #     )
        

    # def forward(self, sender_input, label, receiver_input=None, aux_input=None, epoch=0):
    #     if self.isOptimized:
    #         self.forward_optimized(sender_input, receiver_input, aux_input, epoch)
    #     else:
    #         self.forward_normal(sender_input, label, receiver_input, aux_input, epoch)

    # def forward_optimized(self, sender_input, receiver_input, aux_input, epoch): #actually might not need to have a seperate optimized forward
    #     sender, receiver = self.population.get_pair()
    #     message, sender_aux = sender(sender_input, mode="s")
    #     angle = torch.empty(1, device=message.device).uniform_(-15, 15).item()
    #     if self.training:
    #         message = rotate(
    #             message,
    #             angle=angle,
    #             interpolation=InterpolationMode.BILINEAR,
    #         )
        

    # # def get_sender(self):
    # #     return self.sender

    def forward(self, sender_input, labels, receiver_input, aux_input, epoch):
        # print("reciever_input shpae:", receiver_input.shape, flush=True)
        sender, receiver = self.population.get_pair()
        message, sender_aux = sender(sender_input, mode="s")
        angle = torch.empty(1, device=message.device).uniform_(-15, 15).item()
        # if self.training:
        #     message = rotate(
        #         message,
        #         angle=angle,
        #         interpolation=InterpolationMode.BILINEAR,
        #     )
        message = rotate(
            message,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
        )
        receiver_output, receiver_aux = receiver(
            message, receiver_input, mode="r"
        )
        if self.isOptimized:
            labels = receiver_aux["labels"]

        loss, aux_info = self.loss(
            receiver_output, labels, sender_aux['mu'], sender_aux['log_var'], epoch=epoch,
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            sender_output=message,
            # edge_penalty=aux_info["edge_penalty"],
            vgg_features=sender_aux["vgg_features"],
            receiver_features=receiver_aux["receiver_features"],
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
    

