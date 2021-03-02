'''
Created on Nov 29, 2016

@author: rgrzeszi & jlenk
'''
import numpy as np


class ActivityAugmentation(object):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''

    @staticmethod
    def _interpolate_values(sample):
        '''
        Randomized re-sampling using a linear interpolation
        '''

        resampling_pts = np.random.rand(sample.shape[0]) * (sample.shape[0] - 1)  # Max idx is not shape but shape-1
        resampling_pts = np.sort(resampling_pts)
        resampled = np.zeros(sample.shape)
        for r_i, rsp in enumerate(resampling_pts):
            val0 = int(np.floor(rsp))
            val1 = int(np.ceil(rsp))
            ratio = rsp - val0
            resampled[r_i] = sample[val0] + ((sample[val1] - sample[val0]) * ratio)
        return resampled

    @staticmethod
    def _get_augmented_batch(databatch, std_noise_percentage=0.00):

        augmented_batch = np.copy(databatch)
        if std_noise_percentage == 0 or databatch.shape[0] == 0:
            return augmented_batch

        # Add one percent noise - req. sensor range
        '''
        norm_range = np.max(augmented_batch.reshape(augmented_batch.shape[0] * augmented_batch.shape[1], 
                                                    augmented_batch.shape[2]), axis=0) 
        norm_range -= np.min(augmented_batch.reshape(augmented_batch.shape[0] * augmented_batch.shape[1], 
                                                     augmented_batch.shape[2]), axis=0)
        '''

        for s_i, sample in enumerate(augmented_batch):
            # dev = np.std(sample, axis=0)
            # possibly for each channel separate
            interpolated_sample = ActivityAugmentation._interpolate_values(sample)
            for c_i in np.arange(sample.shape[1]):
                noise = np.random.normal(loc=0.0, scale=std_noise_percentage, size=augmented_batch.shape[1])
                augmented_batch[s_i, :, c_i] = interpolated_sample[:, c_i] + noise
        return augmented_batch

    @staticmethod
    def augment_by_ratio(train_arrays, train_labels, labels_set, min_sample_ratio=0.1):
        '''
        @param train_arrays: array containing the train data [N x D]
        @param train_labels: array containing the labels [N]
        @param labels_set: list of all labels
        @param min_sample_ratio: Classes with less than this percentage of the largest class
                                 are augmented until they meet this requirement.
        '''

        #
        # Create batches of train indices
        # Augment more samples for rare classes
        #
        batch_train_idx = {}
        max_label_count = np.max(np.bincount(train_labels)[labels_set])
        # Iterate all training labels
        for l_i in labels_set:
            # Get all idx where class label equals l_i
            batch_train_idx[l_i] = np.where(train_labels == l_i)[0]
            if batch_train_idx[l_i].shape[0] == 0:
                continue

            train_data_li = train_arrays[batch_train_idx[l_i]]
            # print "Antes de augmentation ", l_i, batch_train_idx[l_i].shape[0]
            # print "Max count", max_label_count, max_label_count*min_sample_ratio
            while batch_train_idx[l_i].shape[0] < (max_label_count * min_sample_ratio):
                # Compute length for augmented batch
                diff_len = int(max_label_count * min_sample_ratio - batch_train_idx[l_i].shape[0]) + 1
                # get augm batch
                augmented_batch = ActivityAugmentation._get_augmented_batch(train_data_li)
                new_labels = np.zeros(augmented_batch.shape[0], dtype=np.int32) + l_i
                # Shorten augmented batch if neccessary
                if augmented_batch.shape[0] > diff_len:
                    augmented_batch = augmented_batch[:diff_len]
                    new_labels = new_labels[:diff_len]
                # Append augmented batch and labels
                train_arrays = np.concatenate((train_arrays, augmented_batch))
                train_labels = np.concatenate((train_labels, new_labels))
                # update batch train_idx
                batch_train_idx[l_i] = np.where(train_labels == l_i)[0]
            # Shuffle idx of each class
            np.random.shuffle(batch_train_idx[l_i])
        return batch_train_idx, train_arrays, train_labels

    @staticmethod
    def get_balanced_random_indices(labels_set, batch_train_idx, train_labels, balanced_batch_size=50):

        # Get initial sample count
        sample_count = 0
        for l_i in labels_set:
            sample_count += batch_train_idx[l_i].shape[0]

        # Compute batch ratio
        batch_train_ratios = {}
        for l_i in labels_set:
            batch_train_ratios[l_i] = float(batch_train_idx[l_i].shape[0]) / sample_count

        # Take always batches that are balanced according to ratio
        random_indices = []
        # Iterate the number of possible batches
        for _ in np.arange(0, sample_count, balanced_batch_size):
            random_idx_i = []
            for l_i in labels_set:
                # Compute percentage of class idx to take
                idx_to_take = int(np.round(balanced_batch_size * batch_train_ratios[l_i], 0))
                idx_to_take = max((1, idx_to_take))
                # Take the class idx
                class_idx = np.copy(batch_train_idx[l_i][:idx_to_take])
                random_idx_i.extend(class_idx)
                # Shorten the arrays
                batch_train_idx[l_i] = batch_train_idx[l_i][idx_to_take:]
            # flatten batch and ramdomize it
            random_idx_i = np.array(random_idx_i, dtype=np.int64)
            np.random.shuffle(random_idx_i)
            # Build random train idx for training
            random_indices.extend(random_idx_i)
        # Make random indices a flatten array
        random_indices = np.array(random_indices, dtype=np.int64)
        return random_indices

    @staticmethod
    def augment_by_number(train_arrays, train_labels, labels_set, number_target_samples=60000):
        '''
        '''
        # Get initial sample count and create initial idx per label
        batch_train_idx = {}
        sample_count = 0
        for l_i in labels_set:
            batch_train_idx[l_i] = np.where(train_labels == l_i)[0]
            sample_count += batch_train_idx[l_i].shape[0]

        sample_diff = number_target_samples - sample_count

        # Compute batch ratio
        if sample_diff > 0:
            # Compute how many samples need to be augmented
            batch_train_ratios = {}
            for l_i in labels_set:
                batch_train_ratios[l_i] = int(batch_train_idx[l_i].shape[0] * sample_diff / float(sample_count))
                # Make sure to augment at least one sample if possible :)
                batch_train_ratios[l_i] += 1

            # Iterate all training labels
            for l_i in labels_set:
                #
                # take random idx for augmentation
                #
                batch_train_idx[l_i] = np.where(train_labels == l_i)[0]
                samples_for_aug = batch_train_ratios[l_i]
                databatch_idx = batch_train_idx[l_i]

                if databatch_idx.shape[0] == 0:
                    continue

                # Shuffle idx and take exactly the number required for augmentation
                np.random.shuffle(databatch_idx)
                while databatch_idx.shape[0] < samples_for_aug:
                    databatch_idx = np.concatenate((databatch_idx, databatch_idx))
                databatch_idx = databatch_idx[:samples_for_aug]

                # get augmented batch
                augmented_batch = ActivityAugmentation._get_augmented_batch(train_arrays[databatch_idx])

                # Append augmented batch and labels
                train_arrays = np.concatenate((train_arrays, augmented_batch))
                new_labels = np.zeros(augmented_batch.shape[0], dtype=np.int32) + l_i
                train_labels = np.concatenate((train_labels, new_labels))
                # Update sample count
                sample_count += samples_for_aug
                # update batch train_idx
                batch_train_idx[l_i] = np.where(train_labels == l_i)[0]
                # Shuffle idx of each class
                np.random.shuffle(batch_train_idx[l_i])

        return batch_train_idx, train_arrays, train_labels