from data.dataset import VOCBboxDataset, TrainDataset, TestDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import os


class FederatedDatasetGenerator:
    @staticmethod
    def setup_malicious_clients(config, train_dataloaders):
        client_indices = list(range(config.num_clients))
        if config.benign:
            return client_indices, []

        num_malicious_clients = int(config.num_clients * config.frac_malicious)
        np.random.seed(config.random_seed)
        malicious_client_indices = set(np.random.choice(client_indices, size=num_malicious_clients, replace=False))
        benign_client_indices = set(client_indices).difference(set(malicious_client_indices))
        assert len(malicious_client_indices.union(benign_client_indices)) == config.num_clients
        for i in range(len(train_dataloaders)):
            train_dataloaders[i].dataset.db.is_benign = (i in benign_client_indices)
        return benign_client_indices, malicious_client_indices

    @staticmethod
    def prepare(config, rand=True):
        return FederatedDatasetGenerator.prepare_IID(config, rand=rand)

    @staticmethod
    def prepare_IID(config, rand=True):
        examples = []
        if config.dataset == 'voc':
            examples += [(config.root_dir_voc, id_.strip()) for id_ in
                         open(os.path.join(config.root_dir_voc, 'ImageSets/Main/trainval.txt'))]
        if config.dataset == 'inria':
            examples += [(config.root_dir_inria, id_.strip()) for id_ in
                         open(os.path.join(config.root_dir_inria, 'ImageSets/Main/trainval.txt'))]
        if rand:
            random.seed(config.random_seed)
            random.shuffle(examples)

        div = len(examples) / float(config.num_clients)
        partitions = [examples[int(round(div * i)): int(round(div * (i + 1)))] for i in range(config.num_clients)]
        assert len(partitions) == config.num_clients, "len(partitions) != opt.num_clients"
        train_dataloaders = FederatedDatasetGenerator._prepare_train_dataloaders(config, partitions)
        test_dataloader = FederatedDatasetGenerator._prepare_test_dataloader(config)
        return test_dataloader, train_dataloaders

    @staticmethod
    def _prepare_train_dataloaders(config, partitions):
        return [DataLoader(TrainDataset(config, partition), shuffle=True) for partition in partitions]

    @staticmethod
    def _prepare_test_dataloader(config, fname='ImageSets/Main/test.txt'):
        if config.dataset == 'inria':
            examples = [(config.root_dir_inria, id_.strip()) for id_ in
                        open(os.path.join(config.root_dir_inria, fname))]
        else:
            examples = [(config.root_dir_voc, id_.strip()) for id_ in
                        open(os.path.join(config.root_dir_voc, fname))]
        return DataLoader(TestDataset(config, examples))
