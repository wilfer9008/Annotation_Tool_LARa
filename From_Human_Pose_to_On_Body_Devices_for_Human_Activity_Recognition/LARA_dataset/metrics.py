'''
Created on Aug 7, 2019

@author: fmoya
'''
import numpy as np
import torch
import logging


class Metrics(object):
    '''
    classdocs
    '''

    def __init__(self, config, dev, attributes):
        '''
        Constructor
        '''

        logging.info('            Metrics: Constructor')
        self.config = config
        self.device = dev
        self.attr = attributes

        return

    ##################################################
    ###########  Precision and Recall ################
    ##################################################
    def get_precision_recall(self, targets, predictions):
        precision = torch.zeros((self.config['num_classes']))
        recall = torch.zeros((self.config['num_classes']))

        x = torch.ones(predictions.size())
        y = torch.zeros(predictions.size())

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        for c in range(self.config['num_classes']):
            selected_elements = torch.where(predictions == c, x, y)
            non_selected_elements = torch.where(predictions == c, y, x)

            target_elements = torch.where(targets == c, x, y)
            non_target_elements = torch.where(targets == c, y, x)

            true_positives = torch.sum(target_elements * selected_elements)
            false_positives = torch.sum(non_target_elements * selected_elements)

            false_negatives = torch.sum(target_elements * non_selected_elements)

            try:
                precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

            except:
                # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                #                                                                                                                              false_positives.item(),
                #                                                                                                                              false_negatives.item()))
                continue

        return precision, recall

    ##################################################
    #################  F1 metric  ####################
    ##################################################

    def f1_metric(self, targets, preds):
        # Accuracy
        if self.config['output'] == 'softmax':
            predictions = torch.argmax(preds, dim=1)
        elif self.config['output'] == 'attribute':
            # predictions = torch.argmin(preds, dim=1)
            predictions = self.atts[torch.argmin(preds, dim=1), 0]

        if self.config['output'] == 'softmax':
            precision, recall = self.get_precision_recall(targets, predictions)
        elif self.config['output'] == 'attribute':
            precision, recall = self.get_precision_recall(targets[:, 0], predictions)

        proportions = torch.zeros(self.config['num_classes'])

        if self.config['output'] == 'softmax':
            for c in range(self.config['num_classes']):
                proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
        elif self.config['output'] == 'attribute':
            for c in range(self.config['num_classes']):
                proportions[c] = torch.sum(targets[:, 0] == c).item() / float(targets[:, 0].size()[0])

        logging.info('            Metric:    \nPrecision: \n{}\nRecall\n{}'.format(precision, recall))

        multi_pre_rec = precision * recall
        sum_pre_rec = precision + recall

        multi_pre_rec[torch.isnan(multi_pre_rec)] = 0
        sum_pre_rec[torch.isnan(sum_pre_rec)] = 0

        # F1 weighted
        weighted_f1 = proportions * (multi_pre_rec / sum_pre_rec)
        weighted_f1[np.isnan(weighted_f1)] = 0

        F1_weighted = torch.sum(weighted_f1) * 2

        # F1 mean
        f1 = multi_pre_rec / sum_pre_rec
        f1[torch.isnan(f1)] = 0

        F1_mean = torch.sum(f1) * 2 / self.config['num_classes']

        return F1_weighted.item(), F1_mean.item()

    ##################################################
    #################  Accuracy  ####################
    ##################################################

    def acc_metric(self, targets, predictions):

        # Accuracy
        if self.config['output'] == 'softmax':
            acc = torch.sum(targets == torch.argmax(predictions, dim=1).type(dtype=torch.cuda.FloatTensor))
        elif self.config['output'] == 'attribute':
            acc = torch.sum(targets == torch.argmin(predictions, dim=1).type(dtype=torch.cuda.FloatTensor))
        acc = acc.item() / float(targets.size()[0])

        return acc

    ##################################################
    ################  Acc attr  #####################
    ##################################################

    def metric_attr(self, targets, predictions):
        # logging.info('        Network_User:    Metrics')

        # Accuracy per vector
        acc_vc = torch.sum(targets == torch.round(predictions), dim=1, dtype=torch.float)
        acc_vc = torch.mean(acc_vc / float(targets.size()[1])).item()

        # Accuracy per attr
        acc_atr = torch.sum((targets == torch.round(predictions)), dim=0, dtype=torch.float)
        acc_atr = acc_atr / float(targets.size()[0])

        return acc_vc, acc_atr

    ##################################################
    ###################  metric  ######################
    ##################################################
    def efficient_distance(self, predictions):
        euclidean = torch.nn.PairwiseDistance()

        predictions = predictions.repeat(self.attr.shape[0], 1, 1)
        predictions = predictions.permute(1, 0, 2)

        atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
        atts = atts.type(dtype=torch.cuda.FloatTensor)

        distances = euclidean(predictions[0], atts)
        distances = distances.view(1, -1)
        for i in range(1, predictions.shape[0]):
            dist = euclidean(predictions[i], atts)
            distances = torch.cat((distances, dist.view(1, -1)), dim=0)

        return distances

    ##################################################
    ###################  metric  ######################
    ##################################################

    def metric(self, targets, predictions):
        # logging.info('        Network_User:    Metrics')

        if self.config['output'] == 'attribute':
            predictions = self.efficient_distance(predictions)

        # Accuracy
        targets = targets.type(dtype=torch.FloatTensor)
        targets = targets.to(self.device)
        acc = self.acc_metric(targets, predictions)

        # F1 metrics
        f1_weighted, f1_mean = self.f1_metric(targets, predictions)

        return acc, f1_weighted, f1_mean



