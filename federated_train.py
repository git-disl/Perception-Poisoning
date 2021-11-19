from data.dataset import VOC_BBOX_LABEL_NAMES, INRIA_BBOX_LABEL_NAMES
from utils.io_tool import create_output_dir
from utils.eval_tool import eval
from fl.dataset_generator import FederatedDatasetGenerator
from fl.client import FederatedClient
from fl.server import FederatedServer
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import copy
import time
import os


########################################################################################################################
#                                    Hyperparameters for Federated Object Detection                                    #
########################################################################################################################
parser = argparse.ArgumentParser()

# Simulation parameters
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--benign', action="store_true", default=False)
parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'inria'])
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--num_rounds', type=int, default=200)
parser.add_argument('--num_test_imgs', type=int, default=1000)

# Server-Client
parser.add_argument('--frac_participants', type=float, default=0.10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'momentum'])
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--load_path', type=str, default=None)

# Poisoning parameters
parser.add_argument('--poison', type=str, default='class', choices=['class', 'bbox', 'objn'])
parser.add_argument('--frac_malicious', type=float, default=0.20)
parser.add_argument('--source_class', type=str, default='person', choices=VOC_BBOX_LABEL_NAMES)
parser.add_argument('--bbox_shrinkage', type=float, default=0.10)  # bbox poison
parser.add_argument('--target_class', type=str, default='pottedplant', choices=VOC_BBOX_LABEL_NAMES)  # class poison
parser.add_argument('--alpha', type=float, default=0.60)

# I/O and Logging
parser.add_argument('--root_dir_voc', type=str, default='/research/datasets/VOCdevkit/VOC2007/')
parser.add_argument('--root_dir_inria', type=str, default='/research/datasets/INRIAdevkit/INRIAPerson/')
parser.add_argument('--root_dir_output', type=str, default='./outputs/')

config = parser.parse_args()

########################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

output_folder = create_output_dir(config)
print('Output folder: %s' % output_folder)


if __name__ == '__main__':
    ####################################################################################################################
    # Dataset Preparation
    ####################################################################################################################
    test_dataloader, train_dataloaders = FederatedDatasetGenerator.prepare(config)

    benign_client_indices, malicious_client_indices = FederatedDatasetGenerator.setup_malicious_clients(config, train_dataloaders)
    print('Benign Clients    (%d): %s' % (len(benign_client_indices), str(benign_client_indices)))
    print('Malicious Clients (%d): %s' % (len(malicious_client_indices), str(malicious_client_indices)))

    ####################################################################################################################
    # Model Construction
    ####################################################################################################################
    print('Constructing the federated server...')
    server = FederatedServer(config, benign_client_indices, malicious_client_indices)
    print('Constructing the simulated federated clients...')
    client = FederatedClient(config)

    ####################################################################################################################
    # Federated Learning
    ####################################################################################################################
    for r in range(config.num_rounds):
        start_time = time.time()

        # Select random clients
        participant_indices = server.get_participant_indices()
        benign_participant_indices = participant_indices.difference(malicious_client_indices)
        malicious_participant_indices = participant_indices.intersection(malicious_client_indices)
        print('Benign    Participants (%d): %s' % (len(benign_participant_indices), benign_participant_indices))
        print('Malicious Participants (%d): %s' % (len(malicious_participant_indices), malicious_participant_indices))

        loss = []
        for client_id in tqdm(participant_indices):
            is_malicious = (client_id in malicious_client_indices)

            ############################################################################################################
            # 1) Get the latest weights from the global model
            _state_dict = copy.deepcopy(server.global_model.state_dict())
            client.load(_state_dict)

            # 2) Run training
            local_loss, local_state_dict, local_n_k = client.update(dataloader=train_dataloaders[client_id],
                                                                    num_epochs=server.num_epochs, lr=server.lr)

            # 3) Send the local model to the server
            server.receive(local_state_dict=local_state_dict, n_k=local_n_k)
            ############################################################################################################
            loss.append(local_loss)
        loss = float(np.mean(loss))

        # Server aggregate
        server.aggregate()
        end_time = time.time()

        # Print and log learning process
        console_output = '[Round %d / %d] Loss: %.4f' % (r + 1, config.num_rounds, loss)

        # Evaluate global model
        eval_detection, eval_result = eval(test_dataloader, server.global_model, test_num_imgs=config.num_test_imgs)

        mAP = eval_result['map']
        source_class_id = INRIA_BBOX_LABEL_NAMES.index(config.source_class) if config.dataset == 'inria' else VOC_BBOX_LABEL_NAMES.index(config.source_class)
        AP_source = eval_result['ap'][source_class_id]
        console_output += ' | mAP: %.2f%% | AP_{%s}: %.2f%%' % (mAP * 100, config.source_class, AP_source * 100)
        # 1. Save ckpt (optional)
        if (r + 1) % 100 == 0:
            saved_path = client.save_ckpt(output_folder=output_folder, model=server.global_model, round=r, map=mAP * 100)
            console_output += ' | Saved: %s' % saved_path
        # 2. Save eval (every round)
        with open(os.path.join(output_folder, 'eval', '%d.pkl' % r), 'wb') as o:
            pickle.dump({'detection': eval_detection, 'result': eval_result}, o)
        # 3. Save log (every round)
        with open(os.path.join(output_folder, 'log.csv'), 'a') as o:
            line = '%d,%f,%f,%s,%s,%f,%s\n' % (r, end_time - start_time, loss,
                                               '_'.join(map(str, malicious_client_indices)),
                                               '_'.join(map(str, participant_indices)), mAP,
                                               '_'.join(map(str, list(eval_result['ap']))))
            o.write(line)
        print(console_output)
