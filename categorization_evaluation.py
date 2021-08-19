import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import gridspec
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import operator
import os
import pickle
import copy
import json
import utils
import rcca


class Category(object):
    def __init__(self, start_idx=None, sequence=None):
        """
        Initializes a category.
        :param start_idx: the index where the non-N sequence starts (int)
        :param sequence: sequence of non-N base symbols (string)
        For example: start_idx=1, sequence=ATTG represent the NAT[T>G]NNN mutation category
        """
        self.start_idx = start_idx
        self.sequence = sequence
        self.pri_score = None  # -size
        self.sec_score = None  # span
        self.tert_score = None  # distance from mutation

    def __eq__(self, other):
        if not isinstance(other, Category):
            return NotImplemented
        return self.start_idx == other.start_idx and self.sequence == other.sequence

    def compute_scores(self):
        """
        Computes category's matching score, and updates primary score (category coverage, or size), secondary score
        (length of non-N sequence, or span) and tertiary score (distance between mutated site and farthest non-N base)
         attributes accordingly.
        """
        idx_of_original_base = SEQ_LENGTH // 2 - 1
        pri_score = 1
        for i, base in enumerate(self.sequence):
            n_bases_in_symbol = len(BASE_DICT[base])
            pri_score = pri_score * n_bases_in_symbol
        pri_score = pri_score * (4 ** (SEQ_LENGTH - len(self.sequence)))
        self.pri_score = -pri_score  # -size
        self.sec_score = len(self.sequence)  # span
        self.tert_score = max(idx_of_original_base - self.start_idx, abs(
            (self.start_idx + len(self.sequence)) - idx_of_original_base + 1))  # distance from mutation


class Dataset(object):
    def __init__(self, dataset_name=None, optimal_k=None, cosmic_sigs=None):
        """
        Initializes a dataset. Main attributes are the exomic mutation data, genomic mutation and gene expression data.
        :param dataset_name: string that matches the name of the corresponding data files, typically the tumor type
        :param optimal_k: the expected number of mutational signatures in the data (int)
        :param cosmic_sigs: list of COSMIC signature names, that are known to be active in this tumor type (list of strings)
        """
        self.dataset_name = dataset_name
        # paths
        path_to_main_mutation_data = PATH_TO_INPUT + '%s_catalog.csv' % self.dataset_name
        path_to_mutation_data_for_signatures = PATH_TO_INPUT + '%s_wgs_catalog.csv' % self.dataset_name
        path_to_ge_data = PATH_TO_INPUT + '%s_ge.csv' % self.dataset_name
        # read and determine k data
        self.optimal_k = optimal_k
        self.cosmic_k = len(cosmic_sigs)
        self.k_range = np.arange(2, np.max([self.optimal_k, self.cosmic_k]) + 3)
        self.k_range_for_sig_analysis = [optimal_k]
        self.H_cosmic = COSMIC_SIGNATURES[cosmic_sigs].to_numpy().T
        self.cosmic_sigs = cosmic_sigs

        self.catalogs = {}
        self.transformation_to_cosmic = {}
        self.opp_normalizer = {'wgs': {}, 'wxs': {}}
        self.folds = {}
        self.mutation_data_for_signatures = pd.read_csv(path_to_mutation_data_for_signatures)
        self.ge_data = pd.read_csv(path_to_ge_data, low_memory=False)

        self.gene_sets = MAIN_GENE_SETS
        self.main_mutation_data = pd.read_csv(path_to_main_mutation_data)

        # we make sure that WGS samples used for signature learning are not present in main data, otherwise - delete
        ge_samples = self.ge_data['icgc_specimen_id'].to_numpy()
        all_mutation_data = self.mutation_data_for_signatures.merge(self.main_mutation_data, on='kmer',
                                                                    how='outer').fillna(0).astype(int, errors='ignore')
        # adjust mutation datasets to kmer size
        if SEQ_LENGTH < 10:
            par = (10 - SEQ_LENGTH) // 2
            temp_catalog = all_mutation_data.groupby(all_mutation_data['kmer'].str[par:-par]).sum()
            temp_catalog = temp_catalog.reset_index()
            all_mutation_data = temp_catalog
        # get lists of kmers
        self.all_kmers = copy.deepcopy(all_mutation_data['kmer'].to_numpy())
        for i, cat in enumerate(self.all_kmers):
            self.all_kmers[i] = cat.replace('[', '').replace(']', '').replace('>', '')
        # split samples to sig and cca
        all_indices = np.arange(all_mutation_data.shape[1]-1)
        all_samples = all_mutation_data.columns[1:]
        ge_sample_indices = np.where(np.in1d(all_samples, ge_samples))[0]
        self.cca_sample_indices = ge_sample_indices
        self.sig_sample_indices = np.setdiff1d(all_indices, ge_sample_indices)
        # adjust ge data
        curr_samples = all_samples[self.cca_sample_indices]
        self.ge_data = self.ge_data.loc[self.ge_data['icgc_specimen_id'].isin(curr_samples)]
        n_samples = self.ge_data.shape[0]
        for repeat in np.arange(REPEATS):
            indices_for_cv = np.random.permutation(n_samples)
            self.folds[repeat] = np.array_split(indices_for_cv, N_FOLDS)
        self.all_mutation_data = all_mutation_data

    def __create_standard_catalog(self):
        """
        Creates the standard mutation catalog (mutation count matrix of size n samples on 96 standard categories).
        :return: standard mutation catalog (n x 96)
        :return: dictionary that maps mutation data k-mers to their matching standard category.
        """
        temp_data = self.all_mutation_data.copy()
        m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA
        assignment = temp_data.groupby(temp_data['kmer'].str[m - 2:m + 5]).ngroup().to_numpy()
        standard_catalog = []
        for i in np.arange(N_CATEGORIES):
            one_category = temp_data.loc[assignment == i].sum(axis=0).to_numpy()[1:]
            standard_catalog.append(one_category)
        standard_catalog = np.array(standard_catalog).T
        temp_data['cluster'] = assignment
        kmer_to_cat = dict(temp_data.groupby('cluster')['kmer'].apply(list))
        return standard_catalog, kmer_to_cat

    def __create_random_catalog(self):
        """
        Creates a random mutation catalog (mutation count matrix of size n samples on N_CATEGORIES random groups).
        Randomly clusters k-mers to N_CATEGORIES groups.
        :return: random mutation catalog (n x N_CATEGORIES)
        :return: dictionary that maps mutation k-mers to their random cluster.
        """
        temp_data = self.all_mutation_data.copy()
        assignment = np.random.choice(np.arange(N_CATEGORIES), size=temp_data.shape[0], replace=True)
        random_catalog = []
        for i in np.arange(N_CATEGORIES):
            one_category = temp_data.loc[assignment == i].sum(axis=0).to_numpy()[1:]
            random_catalog.append(one_category)
        random_catalog = np.array(random_catalog).T
        temp_data['cluster'] = assignment
        kmer_to_cat = dict(temp_data.groupby('cluster')['kmer'].apply(list))
        return random_catalog, kmer_to_cat

    def __check_match(self, kmer, category):
        """
        Checks if a k-mer matches a mutation category
        :param kmer: nucleotide sequence
        :param category: mutation category (category type object)
        :return: True if there is a match, otherwise False
        """
        for i, base in enumerate(category.sequence):
            if kmer[i + category.start_idx] not in BASE_DICT[base]:
                return False
        return True

    def __find_best_match(self, rel_category_indices):
        """
        Finds the best matching category out of several categories that match a mutation sequence, according to the
        three criteria (primary, secondary, tertiary scores).
        :param rel_category_indices: a list of mutation categories that all match a certain k-mer
        :return: 1. True if a single best matching category was identified, otherwise False.
        2. the best matching category or categories.
        """
        rel_category_indices = np.array(rel_category_indices)
        rel_categories = [category for i, category in enumerate(DDC_CATEGORIES) if i in rel_category_indices]
        pri_scores = [category.pri_score for category in rel_categories]
        best_pri_score = np.max(pri_scores)
        indices_of_best = np.nonzero(pri_scores == best_pri_score)[0]
        if len(indices_of_best) == 1:
            return True, rel_category_indices[indices_of_best[0]]
        else:
            rel_category_indices = rel_category_indices[indices_of_best]
            rel_categories = [category for i, category in enumerate(rel_categories) if i in indices_of_best]
            sec_scores = [category.sec_score for category in rel_categories]
            best_sec_score = int(max(sec_scores))
            indices_of_best = np.nonzero(np.array(sec_scores, dtype=int) == best_sec_score)[0]
            if len(indices_of_best) == 1:
                return True, rel_category_indices[indices_of_best[0]]
            else:
                rel_category_indices = rel_category_indices[indices_of_best]
                rel_categories = [category for i, category in enumerate(rel_categories) if i in indices_of_best]
                tert_scores = [category.tert_score for category in rel_categories]
                best_tert_score = np.min(tert_scores)
                indices_of_best = np.nonzero(tert_scores == best_tert_score)[0]
                if len(indices_of_best) == 1:
                    return True, rel_category_indices[indices_of_best[0]]
                else:
                    return False, rel_category_indices[indices_of_best]

    def __create_our_catalog(self):
        """
        Creates the DDC mutation catalog (mutation count matrix of size n samples on N_CATEGORIES DDC categories).
        First performs an assignment of k-mers to m non-disjoint DDC categories, using '__find_best_match' function.
        :return: DDC mutation catalog (n x N_CATEGORIES)
        :return: dictionary that maps mutation k-mers to their best matching DDC category.
        """
        temp_data = self.all_mutation_data.copy()
        assignment = []
        count_conflicts = 0
        empty_category_idx = -1
        for j, category in enumerate(DDC_CATEGORIES):
            if category == EMPTY_CATEGORY:
                empty_category_idx = j
                break
        for i, kmer in enumerate(self.all_kmers):
            rel_cat_indices = []
            for j, category in enumerate(DDC_CATEGORIES):
                if j == empty_category_idx:
                    continue
                kmer_fits_category = self.__check_match(kmer, category)
                if kmer_fits_category:  # potential category <=> no mismatch in category
                    rel_cat_indices.append(j)
            if len(rel_cat_indices) == 0:
                assignment.append(empty_category_idx)
            elif len(rel_cat_indices) == 1:
                assignment.append(rel_cat_indices[0])
            else:
                flag_success, indices_of_best_by_criteria = self.__find_best_match(rel_cat_indices)
                if flag_success:
                    assignment.append(indices_of_best_by_criteria)
                else:
                    count_conflicts += 1
                    np.random.shuffle(indices_of_best_by_criteria)
                    assignment.append(indices_of_best_by_criteria[0])
        print("Assignment completed with %d conflicts" % count_conflicts)
        our_catalog = []
        for i in np.arange(N_CATEGORIES):
            one_category = temp_data.loc[assignment == i].sum(axis=0).to_numpy()[1:]
            our_catalog.append(one_category)
        our_catalog = np.array(our_catalog).T
        temp_data['cluster'] = assignment
        kmer_to_cat = dict(temp_data.groupby('cluster')['kmer'].apply(list))
        return our_catalog, kmer_to_cat

    def __get_normed_7mer_contents(self, kmer_to_category_dicts):
        """
        Computes mutation opportunity normalizers the dataset, according to the input WGS/WXS sequence count data
        and the dataset's k-mer to category content.
        :param kmer_to_category_dicts: dictionaries that map mutation k-mers to their matching category.
        """
        m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA
        for categorization_type, kmer_to_category_dict in kmer_to_category_dicts.items():
            contents = np.zeros((96, 4**7//2))
            for i, category in enumerate(kmer_to_category_dict.keys()):
                kmers = kmer_to_category_dict[category]
                if len(kmers) == 0:
                    continue
                for kmer in kmers:
                    clean_kmer = kmer[:m-1]+kmer[m]+kmer[m+4:]
                    j = KMER_TO_IDX[clean_kmer]
                    contents[i][j] += 1
            self.opp_normalizer['wgs'][categorization_type] = np.sum(
                np.multiply(contents, list(WGS_NORMALIZER_7mer.values())), axis=1)
            self.opp_normalizer['wxs'][categorization_type] = np.sum(
                np.multiply(contents, list(WXS_NORMALIZER_7mer.values())), axis=1)

    def __get_transformation_matrix(self, kmer_to_category_dicts):
        """
        Computes transformation matrices -- from a set of categories to the standard categories.
        :param kmer_to_category_dicts: dictionaries that map mutation k-mers to their matching category.
        """
        m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA
        for categorization_type, kmer_to_category_dict in kmer_to_category_dicts.items():
            content_matrix = np.zeros((N_CATEGORIES, 96), dtype=np.float64)
            for i, category in enumerate(kmer_to_category_dict.keys()):
                kmers = kmer_to_category_dict[category]
                if len(kmers) == 0:
                    continue
                for j, standard_category in enumerate(COSMIC_CATEGORIES):
                    content_matrix[i][j] = np.sum([x[m - 2:m + 5] == standard_category for x in kmers]) / len(kmers)
            self.transformation_to_cosmic[categorization_type] = content_matrix

    def create_all_catalogs(self):
        """
        Creates all the different mutation catalogs (standard, random, DDC) with respect to this dataset.
        All the catalogs consist of n (samples) row vectors, with the same number of total mutations.
        The catalogs differ in the distribution of mutations across each vector, as they are defined by different sets
        of mutation categories.
        """
        print("Assigning k-mers and creating catalogs...")
        full_catalogs = {}
        kmer_to_category_dicts = {}
        full_catalogs['standard'], kmer_to_category_dicts['standard'] = self.__create_standard_catalog()
        full_catalogs['DDC'], kmer_to_category_dicts['DDC'] = self.__create_our_catalog()
        full_catalogs['random'], kmer_to_category_dicts['random'] = self.__create_random_catalog()
        self.__get_normed_7mer_contents(kmer_to_category_dicts)
        self.__get_transformation_matrix(kmer_to_category_dicts)
        for categorization_type, full_catalog in full_catalogs.items():
            self.catalogs[categorization_type] = {}
            full_catalog = normalize(full_catalog, norm='l1', axis=1)
            self.catalogs[categorization_type]['sigs'] = full_catalog[self.sig_sample_indices]
            self.catalogs[categorization_type]['cca'] = full_catalog[self.cca_sample_indices]
        self.catalogs['standard-cosmic'] = self.catalogs['standard']
        print("%s samples:: main: %d, WGS: %d" % (self.dataset_name, self.catalogs['standard']['cca'].shape[0],
                                                  self.catalogs['standard']['sigs'].shape[0]))


class DatasetEvaluation(object):
    def __init__(self, dataset=None):
        """
        Initializes a DatasetEvaluation object. Given a dataset objects, allows the computation and storage of
        mutational signatures, exposures, factorization reconstruction errors, as well as evaluation of signatures'
        similarity to COSMIC and exposures' correlation to different gene sets.
        :param dataset: Dataset object
        """
        self.dataset = dataset
        self.nmf_model_from_wgs = {}
        self.all_wgs_nmf_models = {}
        self.signature_similarities = {}
        self.main_exposures = {}
        self.wgs_exposures = {}
        self.H_cosmic_per_k = {}
        self.reconstruction_error = {}
        self.correlations = {}

    def __kl_divergence(self, V, W, H):
        """
        Computes KL-divergence
        :param V: mutation catalog, count matrix (n x m)
        :param W: exposure matrix (n x k)
        :param H: signature matrix (k x m)
        :return: KL-divergence
        """
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = V.ravel()
        indices = X_data > np.finfo(np.float32).eps
        WH_data = WH_data[indices]
        X_data = X_data[indices]
        WH_data[WH_data == 0] = np.finfo(np.float32).eps
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        div = X_data / WH_data
        kld = np.dot(X_data, np.log(div.astype('float')))
        kld += sum_WH - X_data.sum()
        return kld

    def __opportunity_normalization(self, signatures, categorization_type):
        """
        Performs mutation opportunity normalization on unnormalized signatures
        :param signatures: signature matrix (k x m)
        :param categorization_type: name of relevant categorization (string)
        :return: opportunity-normalized signatures
        """
        opp_norm_sigs = np.divide(signatures, self.dataset.opp_normalizer['wgs'][categorization_type], out=np.zeros_like(signatures),
                              where=self.dataset.opp_normalizer['wgs'][categorization_type] != 0) * self.dataset.opp_normalizer['wxs'][
                        categorization_type]
        opp_norm_sigs = normalize(opp_norm_sigs.astype(np.float64), norm='l1', axis=1)
        return opp_norm_sigs

    def __get_best_nmf_model_and_signatures(self, catalog, k, categorization_type, reps=10):
        """
        Runs sklearn's NMF reps times with random weight initialization, chooses the best model in terms of lowest
        reconstruction error, performs l1 normalization on the H matrix to obtain the signature matrix and applies
        mutation opportunity normalization.
        :param catalog: mutation count matrix (n x m) to be factorized
        :param k: number of components, signatures (int)
        :param categorization_type: name of relevant categorization (string), to apply the appropriate opportunity normalization
        :param reps: number of NMF runs (int)
        :return: the best sklearn NMF model
        """
        np.random.seed()
        models, res = [], []
        for i in np.arange(reps):
            model = NMF(n_components=k, init='random', max_iter=1000, solver='mu', beta_loss='kullback-leibler')
            model.fit(catalog)
            res.append(model.reconstruction_err_)
            models.append(model)
        arg_of_best = np.argmin(res)
        best_model = models[arg_of_best]
        H = best_model.components_
        signatures = normalize(H, norm='l1', axis=1)
        signatures = self.__opportunity_normalization(signatures, categorization_type)
        best_model.components_ = signatures
        return best_model, models

    def __get_exposures_for_cosmic(self):
        """
        Given a standard mutation catalog, computes (using NNLS) the exposures to the known set of active COSMIC signatures.
        :param catalog: mutation count matrix (n x m) to be factorized
        :param k: number of components, signatures (int)
        :param categorization_type: name of relevant categorization (string), to apply the appropriate opportunity normalization
        :param reps: number of NMF runs (int)
        :return: the best sklearn NMF model
        """
        model = NMF(n_components=self.dataset.cosmic_k, init='random', max_iter=1000, solver='mu',
                    beta_loss='kullback-leibler')
        model.fit(self.dataset.catalogs['standard-cosmic']['sigs'])
        all_dataset_cosmic = copy.deepcopy(self.dataset.H_cosmic)
        all_dataset_cosmic = self.__opportunity_normalization(all_dataset_cosmic, 'standard')
        model.components_ = all_dataset_cosmic
        exposures_to_all_cosmic = model.transform(self.dataset.catalogs['standard-cosmic']['sigs'])
        exposure_means = np.mean(exposures_to_all_cosmic, axis=0)
        signatures_by_prevalence = np.argsort(exposure_means)[::-1]
        self.H_cosmic_per_k['standard-cosmic'] = {}
        all_k_exposures = {}
        for k in self.dataset.k_range:
            name_of_k = k
            current_H = all_dataset_cosmic[signatures_by_prevalence[:k]]
            self.H_cosmic_per_k['standard-cosmic'][k] = current_H
            if k > current_H.shape[0]:
                k = current_H.shape[0]
            model = NMF(n_components=k, init='random', max_iter=1000, solver='mu',
                        beta_loss='kullback-leibler')
            model.fit(self.dataset.catalogs['standard-cosmic']['cca'])
            model.components_ = current_H
            all_k_exposures[name_of_k] = model.transform(self.dataset.catalogs['standard-cosmic']['cca'])
        return all_k_exposures

    def get_signatures_from_wgs(self):
        """
        Extracts signatures from all the catalogs.
        """
        print("Computing signatures using WGS...")
        for categorization_type in CATEGORIZATION_TYPES:
            self.nmf_model_from_wgs[categorization_type] = {}
            self.all_wgs_nmf_models[categorization_type] = {}
            for k in self.dataset.k_range:
                self.nmf_model_from_wgs[categorization_type][k], \
                self.all_wgs_nmf_models[categorization_type][k] = self.__get_best_nmf_model_and_signatures(
                    self.dataset.catalogs[categorization_type]['sigs'], k, categorization_type)

    def compute_all_exposures_and_reconstruction_error(self):
        """
        Computes all exposures and reconstruction errors given all catalogs and signatures.
        """
        print("Computing exposures and reconstruction errors...")
        for categorization_type in CATEGORIZATION_TYPES:
            self.main_exposures[categorization_type] = {}
            self.wgs_exposures[categorization_type] = {}
            self.reconstruction_error[categorization_type] = {}
            for k in self.dataset.k_range:
                self.main_exposures[categorization_type][k] = self.nmf_model_from_wgs[categorization_type][k].transform(
                    self.dataset.catalogs[categorization_type]['cca'])
                self.wgs_exposures[categorization_type][k] = self.nmf_model_from_wgs[categorization_type][
                    k].transform(self.dataset.catalogs[categorization_type]['sigs'])
                V = self.dataset.catalogs[categorization_type]['cca']

                self.reconstruction_error[categorization_type][k] = []
                for i_model, cur_model in enumerate(self.all_wgs_nmf_models[categorization_type][k]):
                    W = cur_model.transform(V)
                    H = cur_model.components_
                    self.reconstruction_error[categorization_type][k].append(self.__kl_divergence(V, W, H))

        self.main_exposures['standard-cosmic'] = self.__get_exposures_for_cosmic()
        self.reconstruction_error['standard-cosmic'] = {}
        for k in self.dataset.k_range:
            V = self.dataset.catalogs['standard-cosmic']['cca']
            W = self.main_exposures['standard-cosmic'][k]
            H = self.H_cosmic_per_k['standard-cosmic'][k]
            self.reconstruction_error['standard-cosmic'][k] = self.__kl_divergence(V, W, H)

    def compute_signature_similarities(self):
        """
        Calculates the cosine similarity between each of the extracted DDC signatures and all the COSMIC signatures.
        """
        print("Computing signature similarities to COSMIC...")
        for k in self.dataset.k_range_for_sig_analysis:
            self.signature_similarities[k] = {}
            cur_signatures = copy.deepcopy(self.nmf_model_from_wgs['DDC'][k].components_)
            cur_signatures = np.dot(cur_signatures, self.dataset.transformation_to_cosmic['DDC'])
            similarity_matrix = cosine_similarity(cur_signatures, COSMIC_SIGNATURES_NPY)
            for j in np.arange(similarity_matrix.shape[0]):
                top_max_idx = np.argsort(similarity_matrix[j])[-3:]  # top 3
                max_sigs = list(np.array(SIGNATURE_NAMES)[top_max_idx])
                max_sims = list(similarity_matrix[j][top_max_idx])
                self.signature_similarities[k][j] = list(zip(max_sigs, max_sims))

    def __compute_correlation(self, cca, exposures, gene_expression):
        """
        Computes pearson's correlation between cca-projected exposures and gene expression data.
        :param cca: rcca object
        :param exposures: relevant exposure data
        :param gene_expression: relevant gene expression data
        """
        X = np.dot(exposures, cca.ws[0])
        Y = np.dot(gene_expression, cca.ws[1])
        correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
        return correlation

    def compute_signature_specific_correlations(self):
        """
        Computes - for each DDC signature - the correlation between its corresponding exposures vector and the
        gene expression levels of the different DDR_GENE_SUBSETS.
        """
        print("Computing signature correlation to DDR subsets...")
        folds = self.dataset.folds
        for k in self.dataset.k_range_for_sig_analysis:
            sim_data = self.signature_similarities[k]
            for i_sig in np.arange(len(sim_data)):
                sig_name = "DDC_sig" + str(len(NEW_SIGS) + 1)
                NEW_SIGS[sig_name] = {'signature': self.nmf_model_from_wgs['DDC'][k].components_[i_sig]}
                rel_exposures = self.main_exposures['DDC'][k].T[i_sig].T
                subset_correlations = {}
                for ddr_subset in DDR_GENE_SUBSETS.keys():
                    subset_correlations[ddr_subset] = []
                    rel_genes = DDR_GENE_SUBSETS[ddr_subset]
                    rel_eti_data = self.dataset.ge_data[np.intersect1d(self.dataset.ge_data.columns,
                                                                       rel_genes)].to_numpy()
                    for repeat in np.arange(REPEATS):
                        for cca_fold in np.arange(len(folds[repeat])):
                            rel_indices_train = np.concatenate([folds[repeat][f]
                                                                for f in np.arange(len(folds[repeat]))
                                                                if f != cca_fold])
                            rel_indices_val = folds[repeat][cca_fold]

                            cca = rcca.CCA(reg=1e-4, numCC=1, verbose=False)
                            not_nan_rows = ~np.isnan(rel_eti_data[rel_indices_train]).any(axis=1)
                            cca.train([rel_exposures[rel_indices_train][not_nan_rows].reshape(-1, 1),
                                       rel_eti_data[rel_indices_train][not_nan_rows]])
                            not_nan_rows = ~np.isnan(rel_eti_data[rel_indices_val]).any(axis=1)
                            cur_correlation = self.__compute_correlation(cca, rel_exposures[rel_indices_val][
                                not_nan_rows].reshape(-1, 1), rel_eti_data[rel_indices_val][not_nan_rows])
                            subset_correlations[ddr_subset].append(cur_correlation)

                subset_correlations = {key: np.mean(value) for key, value in subset_correlations.items()}
                subset_correlations_wo_ddr = {key: value for key, value in subset_correlations.items() if key != 'DDR'}
                max_subset, corr = max(subset_correlations_wo_ddr.items(), key=operator.itemgetter(1))
                corr = subset_correlations_wo_ddr[max_subset]
                NEW_SIGS[sig_name]['info'] = {'max_subset': max_subset, 'correlation': corr,
                                              'all_correlations': subset_correlations, 'sim_data': sim_data[i_sig],
                                              'dataset': self.dataset.dataset_name}

    def compute_main_correlations(self, gene_set_name):
        """
        Computes the correlation between each categorization's exposures matrix and the gene expression levels of
        the 'gene_set_name' gene set.
        :param gene_set_name: name of one of the MAIN_GENE_SETS (string)
        """
        gene_set = self.dataset.gene_sets[gene_set_name]
        folds = self.dataset.folds
        rel_ge_data = self.dataset.ge_data[np.intersect1d(self.dataset.ge_data.columns, gene_set)].to_numpy()
        print("Genes in set:", rel_ge_data.shape[1])
        self.correlations[gene_set_name] = {}

        for categorization_type in self.dataset.catalogs.keys():
            self.correlations[gene_set_name][categorization_type] = {}
            for k in self.dataset.k_range:
                all_i_correlations = []
                rep_correlations = []
                rel_exposures = self.main_exposures[categorization_type][k]
                for repeat in np.arange(REPEATS):
                    i_correlations = []
                    for i_cca_fold in np.arange(len(folds[repeat])):
                        rel_indices_train = np.concatenate([folds[repeat][f] for f in np.arange(len(folds[repeat])) if f != i_cca_fold])
                        rel_indices_val = folds[repeat][i_cca_fold]

                        cca = rcca.CCA(reg=1e-4, numCC=1, verbose=False)
                        cca.train([rel_exposures[rel_indices_train], rel_ge_data[rel_indices_train]])
                        corr = self.__compute_correlation(cca, rel_exposures[rel_indices_val],
                                                              rel_ge_data[rel_indices_val])
                        i_correlations.append(corr)
                    rep_correlations.append(i_correlations)
                    all_i_correlations = np.mean(rep_correlations, axis=1)
                self.correlations[gene_set_name][categorization_type][k] = (np.mean(all_i_correlations), np.std(all_i_correlations))
            print(categorization_type, np.mean([value[0] for value in self.correlations[gene_set_name][categorization_type].values()]))


def save_raw_results():
    """
    saves final raw results as python dictionaries.
    """
    for tumor_type in TUMOR_TYPES:
        with open(PATH_TO_OUTPUT + 're_results_%s.pickle' % tumor_type, 'wb') as file:
            pickle.dump(DATASETS_EVALUTATION[tumor_type].reconstruction_error, file)
        with open(PATH_TO_OUTPUT + 'correlation_results_%s.pickle' % tumor_type, 'wb') as file:
            pickle.dump(DATASETS_EVALUTATION[tumor_type].correlations, file)
    with open(PATH_TO_OUTPUT + 'DDC_signatures.pickle', 'wb') as file:
        pickle.dump(NEW_SIGS, file)


def plot_category(frequencies, axs_i, i, categories_prevalence, j=0):
    """
    Helper function for 'produce_categories_figure' function. produces a single category visualization.
    """
    x = 1
    maxi = 0
    if j == 0:
        for freq in frequencies:
            y = 0
            for base, f in freq.items():
                utils.draw(base, x, y, f, axs_i[j])
                y += f
            x += 1
            maxi = max(maxi, y)

        axs_i[j].set_ylabel(i, fontweight='bold', rotation=0, size=5)
        axs_i[j].set_xticks([])
        axs_i[j].set_yticks([])
        axs_i[j].set_xlim((0, x))
        axs_i[j].set_ylim((0, maxi))

    prevalence = categories_prevalence[i - 1] * 100
    axs_i[j + 1].barh([0.5], [prevalence])
    axs_i[j + 1].set_xticks([])
    axs_i[j + 1].set_yticks([])
    axs_i[j + 1].yaxis.set_label_position("right")
    plt.text(0.5, 0.5, '%.1f%%' % prevalence, fontweight='bold', size=7, horizontalalignment='center',
             verticalalignment='center', transform=axs_i[j + 1].transAxes)
    axs_i[j + 1].set_xlim((0, int(np.max(categories_prevalence) * 100)))


def categories_to_vis_convention(cats):
    """
    Helper function for category visualization functions. Switches to a more suitable convention for visualization.
    :param cats: list of categories, functional convention (list of strings)
    :return cats_for_logo: list of categories, visualization convention (list of strings)
    """
    m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA
    cats_for_logo = []
    for i, category in enumerate(cats):
        category = list(category)
        for j, base in enumerate(category):
            if base in BASES[5:]:
                category[j] = "(" + BASE_DICT[base][0] + "/" + BASE_DICT[base][1] + ")"
            else:
                category[j] = base
        category.insert(m - 1, '[')
        category.insert(m + 1, '>')
        category.insert(m + 3, ']')
        cats_for_logo.append(category)
    return cats_for_logo


def produce_categories_figure():
    """
    Produces categories figure and saves to disk as categories.pdf .
    """
    m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA
    padded_cats = ['x' * category.start_idx + category.sequence + 'x' *
                   (SEQ_LENGTH - len(category.sequence) - category.start_idx) for category in DDC_CATEGORIES]
    cats_for_logo = categories_to_vis_convention(padded_cats)

    category_prevalence = {}
    for tumor_type in TUMOR_TYPES:
        category_prevalence[tumor_type] = np.sum(DATASETS_EVALUTATION[tumor_type].dataset.catalogs['DDC']['cca'],
                                                 axis=0) / np.sum(
            DATASETS_EVALUTATION[tumor_type].dataset.catalogs['DDC']['cca'])
    to_sort = np.argsort(np.sum(list(category_prevalence.values()), axis=0))[::-1]

    n_columns = len(TUMOR_TYPES) + 1
    lim = 20

    fig, axs = plt.subplots(lim, n_columns, figsize=(n_columns * 2, lim * 0.3), facecolor='w', edgecolor='k',
                            sharex='col')
    axs[0, 0].set_title('Categories', size=10)

    joker = list(EMPTY_CATEGORY.sequence)
    joker.insert(m - 1, '[')
    joker.insert(m + 1, '>')
    joker.insert(m + 3, ']')

    for c, tumor_type in enumerate(TUMOR_TYPES):
        axs[0, c + 1].set_title('%s' % tumor_type.upper(), size=10)

        idx = 0
        matching_prevalence = category_prevalence[tumor_type][to_sort]
        for cat in np.array(cats_for_logo)[to_sort]:
            if np.array_equal(cat, joker):
                continue
            if idx == lim:
                break
            frequencies = []
            for j, char in enumerate(cat):
                if char in BASES[:5]:
                    frequencies.append({char: 1.})
                elif char in ['[', ']']:
                    continue
                elif char in ['>']:
                    frequencies.append({char: 1.})
                elif char == 'x':
                    frequencies.append({'x': 0.15})
                else:
                    frequencies.append({char[1]: 0.5, char[3]: 0.5})
            plot_category(frequencies, axs[idx], idx + 1, matching_prevalence, j=c)
            idx += 1

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(PATH_TO_OUTPUT + "categories.pdf", bbox_inches='tight')


def produce_reconstruction_error_figure():
    """
    Produces reconstruction error comparison figure and saves to disk as reconstruction_error.pdf .
    """
    fig, axs = plt.subplots(len(TUMOR_TYPES) // FIG_X_IN_A_ROW, FIG_X_IN_A_ROW,
                            figsize=(5 * FIG_X_IN_A_ROW, (len(TUMOR_TYPES) // FIG_X_IN_A_ROW) * 5))
    try: lst_of_axs = axs.ravel()
    except AttributeError: lst_of_axs = [axs]
    artists_for_legend = {}
    for i, tumor_type in enumerate(TUMOR_TYPES):
        for categorization_type in CATEGORIZATION_TYPES:
            re_values_per_k = list(DATASETS_EVALUTATION[tumor_type].reconstruction_error[categorization_type].values())
            mean_values, std_values = np.mean(re_values_per_k, axis=1), np.std(re_values_per_k, axis=1)
            artists_for_legend[categorization_type] = lst_of_axs[i].errorbar(DATASETS[tumor_type].k_range, mean_values,
                                                                             yerr=std_values, label=categorization_type,
                                                                             color=CATEGORIZATION_COLOR_DICT[
                                                                                 categorization_type],
                                                                             linestyle=CATEGORIZATION_LINESTYLE_DICT[
                                                                                 categorization_type])

        lst_of_axs[0].set_ylabel('Reconstruction Error', fontweight='bold')
        lst_of_axs[i].set_xlabel('K (K*=%d)' % DATASETS[tumor_type].optimal_k, fontweight='bold')
        if len(DATASETS[tumor_type].k_range) > 15:
            lst_of_axs[i].set_xticks([v for i, v in enumerate(DATASETS[tumor_type].k_range) if i % 2 != 1])
        else:
            lst_of_axs[i].set_xticks(DATASETS[tumor_type].k_range)
        lst_of_axs[i].set_title("%s (k_cosmic=%d)" % (tumor_type, DATASETS[tumor_type].cosmic_k))
    plt.legend(list(artists_for_legend.values()), labels=CATEGORIZATION_TYPES, loc='upper center',
               bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(PATH_TO_OUTPUT + 'reconstruction_error.pdf')
    plt.show()


def produce_correlation_figure(gene_set_name):
    """
    Produces correlation comparison figure and saves to disk as correlation_to_*gene_set_name*.pdf .
    :param gene_set_name: name of one of the MAIN_GENE_SETS (string)
    """
    fig, axs = plt.subplots(len(TUMOR_TYPES) // FIG_X_IN_A_ROW, FIG_X_IN_A_ROW,
                            figsize=(5 * FIG_X_IN_A_ROW, (len(TUMOR_TYPES) // FIG_X_IN_A_ROW) * 5))
    try: lst_of_axs = axs.ravel()
    except AttributeError: lst_of_axs = [axs]
    artists_for_legend = {}
    for i, tumor_type in enumerate(TUMOR_TYPES):
        for categorization_type in EVALUATED_CATEGORIZATION_TYPES:
            corr_values = list(
                DATASETS_EVALUTATION[tumor_type].correlations[gene_set_name][categorization_type].values())
            mean_values, std_values = [value[0] for value in corr_values], [value[1] for value in corr_values]
            artists_for_legend[categorization_type] = lst_of_axs[i].errorbar(DATASETS[tumor_type].k_range, mean_values,
                                                                             yerr=std_values, label=categorization_type,
                                                                             color=CATEGORIZATION_COLOR_DICT[
                                                                                 categorization_type],
                                                                             linestyle=CATEGORIZATION_LINESTYLE_DICT[
                                                                                 categorization_type])
        lst_of_axs[0].set_ylabel('Correlation', fontweight='bold')
        lst_of_axs[i].set_xlabel('K (K*=%d)' % DATASETS[tumor_type].optimal_k, fontweight='bold')
        if len(DATASETS[tumor_type].k_range) > 15:
            lst_of_axs[i].set_xticks([v for i, v in enumerate(DATASETS[tumor_type].k_range) if i % 2 != 1])
        else:
            lst_of_axs[i].set_xticks(DATASETS[tumor_type].k_range)
        lst_of_axs[i].set_title("%s (k_cosmic=%d)" % (tumor_type, DATASETS[tumor_type].cosmic_k))
    plt.legend(list(artists_for_legend.values()), labels=EVALUATED_CATEGORIZATION_TYPES, loc='upper center',
               bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(PATH_TO_OUTPUT + 'correlation_to_%s.pdf' % gene_set_name)
    plt.show()


def get_our_category_figure_order(categories):
    """
    Helper function for 'produce_ddc_signature_figure' function. Sorts DDC categories as close as possible to the
    standard convention.
    :param categories: list of DDC categories.
    :return categories_new_order: list of indices that sort the categories correctly.
    """
    m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA - 1
    prioritization_of_original = ['C', 'N', 'T']
    prioritization_of_mutation = ['A', 'M', 'R', 'C', 'x', 'N', 'W', 'S', 'Y', 'G', 'K', 'T']
    first_prior = [prioritization_of_original.index(category[m]) for category in categories]
    second_prior = [prioritization_of_mutation.index(category[m+1]) for category in categories]
    overall_prior = list(zip(first_prior, second_prior))
    t = 1
    while t <= m:
        next_left_prior = [prioritization_of_mutation.index(category[m-t]) for category in categories]
        next_right_prior = [prioritization_of_mutation.index(category[m+1+t]) for category in categories]
        overall_prior = [tup + (next_left_prior[i], next_right_prior[i]) for i, tup in enumerate(overall_prior)]
        t += 1
    categories_new_order = [i[0] for i in sorted(enumerate(overall_prior), key=lambda element: element[1])]
    return categories_new_order


def add_categories_to_signature_figure(fig, gs, categories):
    """
    Helper function for 'produce_ddc_signature_figure' function. Adds top categories to signature figure.
    """
    h = 7
    cats_for_logo = categories_to_vis_convention(categories)
    for i_cat, cat in enumerate(cats_for_logo):
        ax = fig.add_subplot(gs[h, 7:9])
        if h == 7:
            ax.set_title('Top Categories')
        frequencies = []
        for j, char in enumerate(cat):
            if char in BASES[:5]:
                frequencies.append({char: 1.})
            elif char in ['[', ']']:
                continue
            elif char in ['>']:
                frequencies.append({char: 1.})
            elif char == 'x':
                frequencies.append({'x': 0.15})
            else:
                frequencies.append({char[1]: 0.5, char[3]: 0.5})
        x = 1
        maxi = 0
        for freq in frequencies:
            y = 0
            for base, f in freq.items():
                utils.draw(base, x, y, f, ax)
                y += f
            x += 1
            maxi = max(maxi, y)
        ax.set_ylabel(i_cat+1, fontweight='bold', rotation=0, color='red', size=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0, x))
        ax.set_ylim((0, maxi))
        h += 1


def produce_ddc_signature_figure():
    """
    Produces DDC signature figures and saves them together as DDC_signatures.pdf .
    """
    mapped_sigs = {sig_name: np.dot(sig_data['signature'],
                                    DATASETS[NEW_SIGS[sig_name]['info']['dataset']].transformation_to_cosmic['DDC'])
                                    for sig_name, sig_data in NEW_SIGS.items()}
    # sort categories for figure
    figure_standard_categories_order = []
    for j in [0, 4, 8, 12, 16, 20]:
        for i in range(4):
            figure_standard_categories_order.extend(list(np.arange(j+24*i, j+24*i+4)))
    figure_standard_categories_order = np.array(figure_standard_categories_order)
    x = np.arange(1, 97)
    xlabels_cosmic = np.array(COSMIC_CATEGORIES)[figure_standard_categories_order]
    xlabels_DDC_padded = ['x' * category.start_idx + category.sequence + 'x' *
                   (SEQ_LENGTH - len(category.sequence) - category.start_idx) for category in DDC_CATEGORIES]
    our_categories_order_for_figure = get_our_category_figure_order(xlabels_DDC_padded)
    m = ORIGINAL_BASE_POSITION_IN_MUTATION_DATA-1
    xlabels_DDC = [category[:m] + '[' + category[m] + '>' + category[m+1] + ']' + category[m+2:]
                    for category in xlabels_DDC_padded]
    xlabels_DDC = np.array(xlabels_DDC)[our_categories_order_for_figure]
    for mapped_sig_name, mapped_sig_i in mapped_sigs.items():
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(12, 8)
        ax1 = fig.add_subplot(gs[0:5, 0:7])
        ax1.set_ylabel('Standard Category Probability (%)')
        ax1.set_xlim(0, 97)
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels_cosmic, rotation=90, size='small')
        ax1.set_ylim(0, 0.15)
        bar_colors = ['cyan'] * 16 + ['black'] * 16 + ['red'] * 16 + ['gray'] * 16 + ['green'] * 16 + ['pink'] * 16
        ax1.bar(x, mapped_sig_i[figure_standard_categories_order], color=bar_colors)

        ax2 = fig.add_subplot(gs[7:12, 0:7])
        ax2.set_xlabel('Category')
        ax2.set_ylabel('DDC Category Probability (%)')
        ax2.set_xlim(0, 97)
        ax2.set_ylim(0, 0.15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(xlabels_DDC, rotation=90, size='small')
        ax2.bar(x, NEW_SIGS[mapped_sig_name]['signature'][our_categories_order_for_figure])

        order_cats_by_prevalence = np.argsort(NEW_SIGS[mapped_sig_name]['signature'])[::-1]
        order_cats_by_prevalence_wo_joker = [idx for idx in order_cats_by_prevalence if xlabels_DDC_padded[idx] != EMPTY_CATEGORY.sequence]
        top_categories = np.array(xlabels_DDC_padded)[order_cats_by_prevalence_wo_joker[:5]]
        add_categories_to_signature_figure(fig, gs, top_categories)

        prev_order = np.argsort(NEW_SIGS[mapped_sig_name]['signature'][our_categories_order_for_figure])[::-1]
        prev_order = [arg for arg in prev_order if xlabels_DDC_padded[arg] != EMPTY_CATEGORY.sequence][:5]
        for i, prev in enumerate(NEW_SIGS[mapped_sig_name]['signature'][our_categories_order_for_figure]):
            if i in prev_order:
                ax2.text(i+1, min(0.125, prev + .01), str(prev_order.index(i)+1), horizontalalignment='center',
                         color='red', fontweight='bold', size=7)

        plt.gcf().subplots_adjust(bottom=0.2, hspace=0.3)
        fig.suptitle('%s - mapped to (%s, %s, %s) with (%.3f, %.3f, %.3f) cosine similarity\nCorrelation with %s = %.3f (DDR=%.3f)' % (
            mapped_sig_name, NEW_SIGS[mapped_sig_name]['info']['sim_data'][2][0],
            NEW_SIGS[mapped_sig_name]['info']['sim_data'][1][0],
            NEW_SIGS[mapped_sig_name]['info']['sim_data'][0][0],
            NEW_SIGS[mapped_sig_name]['info']['sim_data'][2][1],
            NEW_SIGS[mapped_sig_name]['info']['sim_data'][1][1],
            NEW_SIGS[mapped_sig_name]['info']['sim_data'][0][1],
            NEW_SIGS[mapped_sig_name]['info']['max_subset'],
            NEW_SIGS[mapped_sig_name]['info']['correlation'],
            NEW_SIGS[mapped_sig_name]['info']['all_correlations']['DDR']))
        pdf = matplotlib.backends.backend_pdf.PdfPages(PATH_TO_OUTPUT + "DDC_signatures.pdf")
        for fig in np.arange(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()
    plt.cla()
    plt.clf()
    plt.close('all')


if __name__ == "__main__":
    np.random.seed()

    BASES = ['N', 'A', 'G', 'C', 'T', 'W', 'M', 'K', 'R', 'Y', 'S']
    BASE_DICT = {'A': ['A'], 'G': ['G'], 'C': ['C'], 'T': ['T'], 'R': ['A', 'G'], 'Y': ['C', 'T'], 'M': ['A', 'C'],
                 'W': ['A', 'T'], 'S': ['C', 'G'], 'K': ['G', 'T'], 'N': ['A', 'C', 'G', 'T']}
    SEQ_LENGTH = 8
    N_CATEGORIES = 96
    N_FOLDS = 10
    REPEATS = 4
    MAX_FIG_X_IN_A_ROW = 3
    ORIGINAL_BASE_POSITION_IN_MUTATION_DATA = SEQ_LENGTH // 2

    DDR_GENE_SET = np.load('data/DDR_genes.npy', allow_pickle=True)
    HR_GENE_SET = np.load('data/HR_genes.npy', allow_pickle=True)
    MMR_GENE_SET = np.load('data/MMR_genes.npy', allow_pickle=True)
    NER_GENE_SET = np.load('data/NER_genes.npy', allow_pickle=True)
    BER_GENE_SET = np.load('data/BER_genes.npy', allow_pickle=True)
    CGC_ALL_GENE_SET = np.load('data/CGC_genes.npy', allow_pickle=True)

    MAIN_GENE_SETS = {'DDR': DDR_GENE_SET, 'CGC': CGC_ALL_GENE_SET}
    DDR_GENE_SUBSETS = {'DDR': DDR_GENE_SET, 'HR': HR_GENE_SET, 'MMR': MMR_GENE_SET, 'BER': BER_GENE_SET,
                        'NER': NER_GENE_SET}

    EMPTY_CATEGORY = Category(start_idx=0, sequence='N' * SEQ_LENGTH)
    PATH_TO_INPUT = 'data/'
    PATH_TO_OUTPUT = 'results/'
    if not os.path.exists(PATH_TO_OUTPUT):
        os.makedirs(PATH_TO_OUTPUT)

    DDC_CATEGORIES = list(np.load(PATH_TO_INPUT + 'DDC_categorization.npy', allow_pickle=True))

    COSMIC_SIGNATURES = pd.read_csv(PATH_TO_INPUT + 'cosmic_v2_signatures_plus_artifacts.csv')
    SIGNATURE_NAMES = COSMIC_SIGNATURES.columns[3:]
    COSMIC_CATEGORIES = COSMIC_SIGNATURES['Somatic Mutation Type'].to_numpy()
    COSMIC_SIGNATURES_NPY = COSMIC_SIGNATURES.to_numpy().T[3:]

    with open(PATH_TO_INPUT + 'wgs_opportunity_7mer_dictionary.pickle', 'rb') as handle:
        WGS_NORMALIZER_7mer = pickle.load(handle)
    with open(PATH_TO_INPUT + 'wxs_opportunity_7mer_dictionary.pickle', 'rb') as handle:
        WXS_NORMALIZER_7mer = pickle.load(handle)
    KMER_TO_IDX = {kmer: idx for idx, kmer in enumerate(WXS_NORMALIZER_7mer.keys())}

    CATEGORIZATION_TYPES = ['standard', 'DDC', 'random']
    STANDARD_COSMIC_CATEGORIZATION = ['standard-cosmic']

    CATEGORIZATION_COLOR_DICT = {'DDC': 'blue', 'standard': 'orange', 'random': 'green', 'standard-cosmic': 'orange'}
    CATEGORIZATION_LINESTYLE_DICT = {'DDC': 'solid', 'standard': 'solid', 'random': 'solid',
                                     'standard-cosmic': 'dashed'}

    configs = json.load(open('config.json', 'r'))
    DATASETS = {}
    for config_dataset in configs['datasets']:
        DATASETS[config_dataset['dataset_name']] = Dataset(dataset_name=config_dataset['dataset_name'],
                                                           optimal_k=config_dataset['optimal_k'],
                                                           cosmic_sigs=config_dataset['cosmic_signatures'])
    TUMOR_TYPES = list(DATASETS.keys())
    FIG_X_IN_A_ROW = len(TUMOR_TYPES) if len(TUMOR_TYPES) < MAX_FIG_X_IN_A_ROW else MAX_FIG_X_IN_A_ROW

    DATASETS_EVALUTATION = {}
    NEW_SIGS = {}
    for dataset_name, dataset in DATASETS.items():
        print("\n%s evaluation:\n" % dataset_name)
        dataset.create_all_catalogs()
        dataset_evaluation = DatasetEvaluation(dataset)
        DATASETS_EVALUTATION[dataset_name] = dataset_evaluation
        dataset_evaluation.get_signatures_from_wgs()
        dataset_evaluation.compute_all_exposures_and_reconstruction_error()
        dataset_evaluation.compute_signature_similarities()
        dataset_evaluation.compute_signature_specific_correlations()
        for gene_set_name in dataset.gene_sets.keys():
            print("Computing correlations to %s..." % gene_set_name)
            dataset_evaluation.compute_main_correlations(gene_set_name)

    print("Saving results and producing figures...")
    EVALUATED_CATEGORIZATION_TYPES = list(DATASETS_EVALUTATION[TUMOR_TYPES[0]].dataset.catalogs.keys())
    produce_categories_figure()
    produce_reconstruction_error_figure()
    produce_ddc_signature_figure()
    save_raw_results()
    for gene_set_name in DATASETS_EVALUTATION[TUMOR_TYPES[0]].correlations.keys():
        produce_correlation_figure(gene_set_name)
