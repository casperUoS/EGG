import random

import torch
from torch import nn
import torch.nn.functional as F

from egg.core import LoggingStrategy


class SenderWrapper(nn.Module):
    def __init__(self, vision_encoder, sketch_decoder):
        super(SenderWrapper, self).__init__()
        self.vision_encoder = vision_encoder
        self.sketch_decoder = sketch_decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + std*eps, std

    def forward(self, x):
        mu, logvar, sender_aux = self.vision_encoder(x)
        z, std = self.reparameterize(mu, logvar)
        ouput, aux = self.sketch_decoder(z)


class ReceiverWrapper(nn.Module):
    def __init__(self, sketch_encoder, vision_encoder):
        super(ReceiverWrapper, self).__init__()
        self.sketch_encoder = sketch_encoder
        self.vision_encoder = vision_encoder

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h = self.vision_encoder(h)
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h = h.view(h.size(0), -1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h

    def forward(self, signal, x):
        emb = self.return_embeddings(x)

        h_s = self.sketch_encoder(signal)
        # embd_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size

        log_probs = F.log_softmax(out, dim=1)

        auxdata = {
            "h_s": h_s,
        }

        return log_probs, auxdata


class AgentWrapper(nn.Module):
    def __init__(self, sketch_encoder, vision_encoder, sketch_decoder):
        super(AgentWrapper, self).__init__()
        self.sketch_encoder = sketch_encoder
        self.vision_encoder = vision_encoder
        self.sketch_decoder = sketch_decoder

        self.sender = SenderWrapper(sketch_encoder, sketch_decoder)
        self.receiver = ReceiverWrapper(sketch_encoder, vision_encoder)

    def get_sender(self):
        return self.sender

    def get_receiver(self):
        return self.receiver


class Population(nn.Module):
    def __init__(self):
        super(Population, self).__init__()
        self.population_list = []

    def generate_population(self, agent, size):
        for _ in range(size):
            self.add_agent(agent)

    def add_agent(self, agent):
        self.population_list.append(agent)

    def __len__(self):
        return len(self.population_list)

    def get_pair(self):
        if len(self.population_list) < 2:
            raise IndexError('Population must have at least two agents')

        temp_population = self.population_list.copy()
        sender = temp_population.pop(random.randint(0,len(temp_population)-1))
        receiver = temp_population.pop(random.randint(0,len(temp_population)-1))

        return sender, receiver



class PopulationDiffGame(nn.Module):

    def __init__(
            self,
            population,
            loss,
            train_logging_strategy = None,
            test_logging_strategy = None,
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
        super(PopulationDiffGame, self).__init__()
        self.population = population
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, target_position=None, aux_input=None):
        sender, receiver = self.population.get_pair()
        message, sender_aux = sender(sender_input, aux_input)
        receiver_output, receiver_emb = receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, target_position, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            sender_output=message,
            edge_penalty=aux_info["edge_penalty"],
            vgg_features=sender_aux["vgg_features"],
            receiver_features=receiver_emb,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction

    # def get_sender(self):
    #     return self.sender



