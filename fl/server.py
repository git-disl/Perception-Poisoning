from model import FasterRCNNVGG16
import numpy as np
import random
import copy


class FederatedServer(object):
    def __init__(self, config, benign_client_indices=(), malicious_client_indices=()):
        self.config = config
        self.global_model = FasterRCNNVGG16(config).cuda(device=0)
        self.num_clients = config.num_clients
        self.frac_participants = config.frac_participants
        self.num_epochs = config.local_epochs
        self.lr = config.lr
        self.num_participants = int(self.num_clients * self.frac_participants)
        self.benign_client_indices = benign_client_indices
        self.malicious_client_indices = malicious_client_indices
        if len(self.benign_client_indices) == 0:
            self.benign_client_indices = tuple(range(self.num_clients))

        # Private variables for handling model weight streams from clients
        self._global_state_dict = None
        self._N = 0

        # Private variables for momentum
        self._v = None

    def get_participant_indices(self):
        if not self.config.benign and self.config.alpha is not None:
            participant_indices = self._get_participant_indices_with_availability()
        else:
            participant_indices = np.random.permutation(self.num_clients)[:self.num_participants]
        return set(participant_indices)

    def _get_participant_indices_with_availability(self):
        _benign_client_indices = list(copy.deepcopy(self.benign_client_indices))
        _malicious_client_indices = list(copy.deepcopy(self.malicious_client_indices))
        outputs = []
        while len(outputs) != self.num_participants:
            if len(_malicious_client_indices) > 0 and random.uniform(0, 1) < self.config.alpha:
                selected_index = random.choice(_malicious_client_indices)
                outputs.append(selected_index)
                _malicious_client_indices.remove(selected_index)
            else:
                selected_index = random.choice(_benign_client_indices)
                outputs.append(selected_index)
                _benign_client_indices.remove(selected_index)
        return outputs

    def send(self, client):
        client.frcnn.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

    def receive(self, local_state_dict, n_k):
        # Deep copy the model weights due to the pass-by-reference problem
        _local_state_dict = copy.deepcopy(local_state_dict)

        if self._global_state_dict is None:
            # Case 1: The first time receiving model weights in a communication round
            self._global_state_dict = _local_state_dict
            for k in self._global_state_dict.keys():
                self._global_state_dict[k] = n_k * self._global_state_dict[k]
        else:
            # Case 2: Not the first time receiving model weights in a communication round
            for k in self._global_state_dict.keys():
                self._global_state_dict[k] = self._global_state_dict[k] + n_k * _local_state_dict[k]

        # Update the total number of examples involved in the current communication round
        self._N += n_k

    def aggregate(self):
        if self.config.optimizer == 'sgd':
            self._aggregate_sgd()
        elif self.config.optimizer == 'momentum':
            self._aggregate_momentum()

    def _aggregate_sgd(self):
        # Normalize the model weights by the total number of examples to complete the weighted averaging
        for k in self._global_state_dict.keys():
            self._global_state_dict[k] = self._global_state_dict[k] / float(self._N)

        # Register the new weights of the global model
        self.global_model.load_state_dict(copy.deepcopy(self._global_state_dict))

        # Reset all variables for the next communication round
        self._reset()

    def _aggregate_momentum(self, momentum=0.90):
        # Normalize the model weights by the total number of examples to complete the weighted averaging
        for k in self._global_state_dict.keys():
            self._global_state_dict[k] = self._global_state_dict[k] / float(self._N)

        w_old = self.global_model.state_dict()
        w_new = self._global_state_dict

        if self._v is None:
            self._v = {k: -(w_old[k] - w_new[k]) for k in self._global_state_dict.keys()}
        else:
            self._v = {k: (momentum * self._v[k] - (w_old[k] - w_new[k])) for k in self._global_state_dict.keys()}

        theta = {k: (w_old[k] + self._v[k]) for k in self._v.keys()}

        # Register the new weights of the global model
        self.global_model.load_state_dict(theta)

        # Reset all variables for the next communication round
        self._reset()

    def _reset(self):
        self._global_state_dict = None
        self._v = None
        self._N = 0
