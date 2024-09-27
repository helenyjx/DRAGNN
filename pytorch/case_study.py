import torch
import numpy as np
from loader import *
from model import Model
from utils import create_model, get_logger
import evaluation
import scipy.io as scio
import os


def case_result(specific_name):
    path = "./data/Fdataset/Fdataset"
    _, _, drug_name, _, _, _, _ = load_mat(path + ".mat")
    score = scio.loadmat('result/Fdataset/{}.mat'.format(specific_name))
    y_score = score["predict"].reshape(-1)
    for i in range(len(drug_name)):
        drug_name[i] = drug_name[i][0]
    drug2score = {}
    for i in range(len(y_score)):
        drug2score[drug_name[i]] = y_score[i]
    sorted_data = sorted(drug2score.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_potential_drug = {}
    for i in range(len(sorted_data)):
        sorted_potential_drug[sorted_data[i][0]] = sorted_data[i][1]

    path = os.path.join(".", "result/Fdataset/")
    if not os.path.exists(path):
        os.makedirs(path)
    potential_drug_file = "{}-potential-drug.txt".format(specific_name)
    with open(path + potential_drug_file, "w") as f:
        for key, value in sorted_potential_drug.items():
            f.write(str(key) + "\t" + "{}".format(value) + "\n")


def case_study(config):
    log_path = os.path.join(".", config['log_path'])
    logger = get_logger(log_path)

    disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_matrix, all_train_mask, all_test_mask = case_data_reader(
                                                    logger=logger,
                                                    config=config,
                                                    dataset=config['dataset'],
                                                    disease_disease_topk=config['disease_knn_number'],
                                                    drug_drug_topk=config['drug_knn_number'])
    all_scores, all_labels = [], []
    for fold_num in range(len(all_train_mask)):
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        train_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_train_mask[fold_num], train_matrix)
        valid_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], train_matrix)
        test_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], train_matrix)
        train_manager = case_BatchManager(train_data, config['batch_size'], "train")
        valid_manager = case_BatchManager(valid_data, config['batch_size'], 'valid')
        test_manager = case_BatchManager(test_data, config['batch_size'], "test")
        steps_per_epoch = train_manager.len_data

        model = create_model(Model, config, logger)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        logger.info("start training fold {}".format(fold_num + 1))
        loss_list = []
        for epoch in range(config['max_epoch']):
            model.train()
            for batch in train_manager.iter_batch(shuffle=True):
                optimizer.zero_grad()
                loss, _ = model(*batch)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            auroc, aupr = evaluate(model, "valid", valid_manager, logger, all_scores, all_labels, config,
                                   fold_num)
            print("fold {} valid auroc :{:>.5f}".format(fold_num + 1, auroc))
            print("fold {} valid aupr :{:>.5f}".format(fold_num + 1, aupr))
        final_test_auroc, final_test_aupr = evaluate(model, "test", test_manager, logger,
                                                     all_scores,
                                                     all_labels, config, fold_num)
        print("fold {} final test auroc :{:>.5f}".format(fold_num + 1, final_test_auroc))
        print("fold {} final test aupr :{:>.5f}".format(fold_num + 1, final_test_aupr))

    case_result(config['specific_name'])


def evaluate(model, name, data, logger, all_scores, all_labels, config, fold_num=0):
    logger.info("evaluate data:{}".format(name))
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in data.iter_batch():
            loss, z = model(*batch)
            scores.append(z.cpu().numpy())
            labels.append(batch[-1])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    result = evaluation.evaluate(scores, labels)
    auroc = result['auroc']
    aupr = result['aupr']
    if name == "valid":
        return auroc, aupr
    elif name == "test":
        logger.info("fold {} final test auroc :{:>.5f}".format(fold_num + 1, auroc))
        logger.info("fold {} final test aupr :{:>.5f}".format(fold_num + 1, aupr))
        all_scores.append(scores)
        all_labels.append(labels)
        folder = os.path.join('result', "Fdataset")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = "{}.mat".format(config["specific_name"])
        file_path = os.path.join(folder, file_name)
        scio.savemat(file_path, {'label': labels,
                                 'predict': scores})
        return auroc, aupr


if __name__ == "__main__":
    case_result("parkinson")
