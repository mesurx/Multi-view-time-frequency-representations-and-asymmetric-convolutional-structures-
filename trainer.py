

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sklearn

from sklearn.mixture import GaussianMixture
from loss import ASDLoss
import utils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']  # 这是什么参数
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(device)
        self.transform = kwargs['transform']

    def train(self, train_loader):
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs # 10
        early_stop_epochs = self.args.early_stop_epochs   # -1
        start_valid_epoch = self.args.start_valid_epoch   # 0
        num_steps = len(train_loader) # 284 =36283/128
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0

        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            total_correct = 0  #
            total_samples = 0
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x_wavs, x_mels,centroids,energys,labels) in train_bar:  #
                # forward
                x_wavs, x_mels, centroids, energys = x_wavs.float().to(device), x_mels.float().to(device),centroids.float().to(device),energys.float().to(device)
                labels = labels.reshape(-1).long().to(device)
                logits, _ = self.net(x_wavs, x_mels, centroids, energys,labels)  # #logits未进行softmax
                loss = self.criterion(logits, labels)
                with torch.no_grad():
                    output = torch.argmax(logits, dim=1)  #
                    correct = (output == labels).sum().item()  #
                    total_correct += correct
                    total_samples += labels.size(0)

                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1


            avg_loss = sum_loss / num_steps

            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')

            if 0 <= epoch:
                avg_auc, avg_pauc = self.test(save=False, gmm_n=False,current_epoch=epoch)      # 这个已经开始测试,是类里面的函数
                self.writer.add_scalar(f'auc', avg_auc, epoch)
                self.writer.add_scalar(f'pauc', avg_pauc, epoch)

                if avg_auc + avg_pauc >= best_metric:  # 保存最佳模型
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, f' {epoch}_best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')

                else: #
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break



    def test(self, save=False, gmm_n=False,current_epoch=None):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if gmm_n:
            result_dir = os.path.join(self.args.result_dir, self.args.version, f'GMM-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)

        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv
            csv_lines.append([machine_type])  #
            csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []


            feature_data = [] if machine_type == 'fan' else None
            headers_initialized = False

            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)  #

            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]  #

                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')  #
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')  #
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]  #

                if gmm_n:
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')  #
                    features = self.get_latent_features(train_files)  # 全连接层前的特征
                    means_init = net.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)  #

                for file_idx, file_path in enumerate(test_files):  #

                    x_wavs, x_mels, centroids, energys,label = self.transform(file_path)
                    x_wavs, x_mels, centroids, energys = x_wavs.unsqueeze(0).float().to(device), x_mels.unsqueeze(0).float().to(device), centroids.unsqueeze(0).float().to(device),energys.unsqueeze(0).float().to(device)
                    label = torch.tensor([label]).long().to(device) #
                    with torch.no_grad():
                        predict_ids, feature = net(x_wavs, x_mels, centroids, energys,label)

                    if machine_type == 'fan' and current_epoch == 4:
                        feature_np = feature.squeeze().detach().cpu().numpy()  #

                        if not headers_initialized:
                            feature_size = feature_np.shape[0]
                            headers = ['filename', 'y_true'] + [f'feat_{i + 1}' for i in range(feature_size)]  #
                            feature_data.append(headers)
                            headers_initialized = True

                        #
                        filename = os.path.basename(file_path)
                        true_label = y_true[file_idx]  #

                        # 
                        row = [filename, true_label] + feature_np.tolist()
                        feature_data.append(row)

                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()  ## softmax
                        y_pred[file_idx] = probs[label]  #

                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)

                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)  ##
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])    #

            """ 新添 """
            if machine_type == 'fan' and len(feature_data) > 0:
                feature_csv_path = os.path.join(result_dir, f'tsne_feature_{machine_type}.csv')
                utils.save_csv(feature_csv_path, feature_data)


            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_p_auc*100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        self.logger.info(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, csv_lines)
        return avg_auc, avg_pauc #


    def evaluator(self, save=True, gmm_n=False):
        result_dir = os.path.join('./evaluator/teams', self.args.version)
        if gmm_n:
            result_dir = os.path.join('./evaluator/teams', self.args.version + f'-gmm-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)

        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.test_dirs), sorted(self.args.add_dirs))):
            machine_type = target_dir.split('/')[-2]
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files = utils.get_filename_list(target_dir, pattern=f'{id_str}*')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                if gmm_n:
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')
                    features = self.get_latent_features(train_files)
                    means_init = net.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    # means_init = None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(device), x_mel.unsqueeze(0).float().to(
                        device)
                    label = torch.tensor([label]).long().to(device)
                    with torch.no_grad():
                        predict_ids, feature = net(x_wav, x_mel, label)
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)

    def get_latent_features(self, train_files):
        pbar = tqdm(enumerate(train_files), total=len(train_files))
        self.net.eval()
        classifier = self.net.module if self.args.dp else self.net
        features = []
        for file_idx, file_path in pbar:
            x_wav, x_mel, label = self.transform(file_path)
            x_wav, x_mel = x_wav.unsqueeze(0).float().to(device), x_mel.unsqueeze(0).float().to(
                device)
            label = torch.tensor([label]).long().to(device)
            with torch.no_grad():
                _, feature, _ = classifier(x_wav, x_mel, label)
            if file_idx == 0:
                features = feature.cpu()
            else:
                features = torch.cat((features.cpu(), feature.cpu()), dim=0)
        if self.args.use_arcface: features = F.normalize(features)
        return features.numpy()


    def fit_GMM(self, data, n_components, means_init=None):
        print('=' * 40)
        print('Fit GMM in train data for test...')
        np.random.seed(self.args.seed)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, reg_covar=1e-3, verbose=2)
        gmm.fit(data)
        print('Finish GMM fit.')
        print('=' * 40)
        return gmm
