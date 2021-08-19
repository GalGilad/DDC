import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import argparse
import pandas as pd
import random
from datetime import timedelta
import time
import pickle
from multiprocessing import Pool
import datetime
import copy
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
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


class Categorization(object):
    def __init__(self, categories=None):
        """
        Initializes a categorization.
        :param categories: set of initial category type objects
        :Attr P_n_bases: distribution of category non-N sequence length
        """
        self.categories = categories
        self.P_n_bases = np.random.dirichlet((1, C_DIRICHLET, 1, 1, 1), 1)[0]
        if self.categories is None:
            self.categories = []
            self.categories = self.__generate_categories(n_generate=N_CATEGORIES)
        elif len(self.categories) < N_CATEGORIES:
            self.categories.extend(self.__generate_categories(n_generate=N_CATEGORIES - len(self.categories)))
        self.updated_nmf = False
        self.opp_normalizer = {'wgs': {}, 'wxs': {}}
        self.nmf_model = {}
        self.assignments = {}
        self.catalogs = {}
        self.kmers_to_categories = {}
        self.train_exposures = {}
        self.fitness_scores = {}
        self.validation_scores = {}

    def __opportunity_normalization(self, signatures, dataset_name):
        """
        Performs mutation opportunity normalization on unnormalized signatures
        :param signatures: signature matrix (k x m)
        :param dataset_name: name of relevant dataset (string), to apply the appropriate opportunity normalization
        :return: opportunity-normalized signatures
        """
        opp_norm_sigs = np.divide(signatures, self.opp_normalizer['wgs'][dataset_name], out=np.zeros_like(signatures),
                                  where=self.opp_normalizer['wgs'][dataset_name] != 0) \
                        * self.opp_normalizer['wxs'][dataset_name]
        opp_norm_sigs = normalize(opp_norm_sigs.astype(np.float64), norm='l1', axis=1)
        return opp_norm_sigs

    def __get_signatures_from_best_nmf_model(self, dataset_name, k, reps=10):
        """
        Runs sklearn's NMF reps times with random weight initialization, chooses the best model in terms of lowest
        reconstruction error, performs l1 normalization on the H matrix to obtain the signature matrix and applies
        mutation opportunity normalization.
        :param dataset_name: name of relevant dataset (string), to apply the appropriate opportunity normalization
        :param k: number of components, signatures (int)
        :param reps: number of NMF runs (int)
        :return: the best sklearn NMF model
        """
        np.random.seed()
        models, res = [], []
        catalog = self.catalogs[dataset_name]['wgs']
        for i in np.arange(reps):
            model = NMF(n_components=k, init='random', max_iter=1000, solver='mu', beta_loss='kullback-leibler')
            model.fit(catalog)
            res.append(model.reconstruction_err_)
            models.append(model)
        arg_of_best = np.argmin(res)
        best_model = models[arg_of_best]
        H = best_model.components_
        signatures = normalize(H, norm='l1', axis=1)
        signatures = self.__opportunity_normalization(signatures, dataset_name)
        best_model.components_ = signatures
        return best_model

    def __generate_categories(self, n_generate):
        """
        Generates a list of random unique categories of self.P_n_bases consecutive non-N sequence lengths,
        with a well defined original base (before mutation)
        :param n_generate: number of categories to generate
        :return: generated categories (list of category type objects)
        """
        np.random.seed()
        generated_categories = []
        if EMPTY_CATEGORY not in self.categories:
            generated_categories.append(EMPTY_CATEGORY)
        sub_loc = SEQ_LENGTH // 2 - 1
        n_bases = np.arange(3, 3 + len(self.P_n_bases))

        while len(generated_categories) != n_generate:
            n_of_bases = np.random.choice(n_bases, size=1, replace=False, p=self.P_n_bases)
            loc_start_min = max(0, sub_loc + 1 - n_of_bases)
            loc_start_max = min(sub_loc, SEQ_LENGTH - n_of_bases)
            start_idx = np.random.choice(np.arange(loc_start_min, loc_start_max + 1), size=1)[0]
            sub_in_sequence_idx = sub_loc - start_idx
            template = np.random.choice(BASES[1:], size=n_of_bases)
            sub_base = np.random.choice(BASES[3:5], size=1)[0]  # only C / T
            template[sub_in_sequence_idx] = sub_base
            if sub_in_sequence_idx != len(template) - 1:  # mutated not as sub base
                template[sub_in_sequence_idx + 1] = \
                np.random.choice([base for base in BASES[1:] if sub_base not in BASE_DICT[base]], size=1)[0]
            sequence = ''.join(template)
            category = Category(start_idx, sequence)
            if category not in generated_categories:
                if category not in self.categories:
                    generated_categories.append(category)
        self.updated_nmf = False
        return generated_categories

    def __check_match(self, kmer, category):
        """
        Checks if a mutation sequence (kmer) matches a mutation category
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
        :param rel_category_indices: a list of mutation categories that all match a certain mutation sequence
        :return: 1. True if a single best matching category was identified, otherwise False.
        2. the best matching category or categories.
        """
        rel_category_indices = np.array(rel_category_indices)
        rel_categories = [category for i, category in enumerate(self.categories) if i in rel_category_indices]
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

    def __get_normalizers(self):
        """
        Computes mutation opportunity normalizers for each dataset, according to the input WGS/WXS sequence count data
        and each dataset's kmer (mutation sequence) to category content.
        """
        m = SEQ_LENGTH // 2
        for dataset_name, kmers_to_categories in self.kmers_to_categories.items():
            contents = np.zeros((96, 4 ** (SEQ_LENGTH-1) // 2))
            for i, category in enumerate(kmers_to_categories.keys()):
                kmers = kmers_to_categories[category]
                if len(kmers) == 0:
                    continue
                for kmer in kmers:
                    clean_kmer = kmer[:m - 1] + kmer[m] + kmer[m + 4:]
                    j = KMER_TO_IDX[clean_kmer]
                    contents[i][j] += 1
            self.opp_normalizer['wgs'][dataset_name] = np.sum(
                np.multiply(contents, list(WGS_OPPORTUNITY.values())), axis=1)
            self.opp_normalizer['wxs'][dataset_name] = np.sum(
                np.multiply(contents, list(WXS_OPPORTUNITY.values())), axis=1)

    def __get_catalogs(self):
        """
        Computes normalized WGS and WXS mutation catalogs (n samples x m categories) for each dataset, according to the
        sequence to category assignments.
        """
        for dataset in DATASETS:
            self.catalogs[dataset.dataset_name] = {}
            catalog = []
            indices_range = np.arange(96)
            for i in indices_range:
                one_category = dataset.mutation_data.loc[self.assignments[dataset.dataset_name] == i].sum(
                    axis=0).to_numpy()[1:]
                catalog.append(one_category)
            catalog = np.array(catalog).T
            catalog = normalize(catalog, norm='l1', axis=1)
            self.catalogs[dataset.dataset_name]['wgs'] = catalog[dataset.sig_sample_indices]
            self.catalogs[dataset.dataset_name]['train'] = catalog[dataset.cca_sample_indices]

            temp_data = dataset.mutation_data.copy()
            temp_data['cluster'] = self.assignments[dataset.dataset_name]
            self.kmers_to_categories[dataset.dataset_name] = dict(temp_data.groupby('cluster')['kmer'].apply(list))

    def __run_nmf(self):
        """
        For each dataset and k in corresponding k_range, run NMF to k components to get signature exposure matrices.
        """
        self.__get_catalogs()
        self.__get_normalizers()
        for dataset in DATASETS:
            self.train_exposures[dataset.dataset_name] = {}
            self.nmf_model[dataset.dataset_name] = {}
            for k in dataset.k_range:
                model = self.__get_signatures_from_best_nmf_model(dataset.dataset_name, k)
                exposures = model.transform(self.catalogs[dataset.dataset_name]['train'])
                self.train_exposures[dataset.dataset_name][k] = exposures
                self.nmf_model[dataset.dataset_name][k] = model
        self.updated_nmf = True

    def __compute_correlation(self, cca, exposures, gene_expression):
        """
        Compute correlation between exposure matrix and gene expression data, based on the canonical correlation
        analysis (coefficients).
        :param cca: canonical correlation analysis result
        :param exposures: exposure matrix (n x k)
        :param gene_expression: expression matrix (n x g)
        :return: correlation
        """
        X = np.dot(exposures, cca.ws[0])
        Y = np.dot(gene_expression, cca.ws[1])
        correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
        return correlation

    def __compute_fitness(self):
        """
        For each dataset, computes the average out of sample correlation between signature exposure and
        gene expression over all k, folds and repeats. CCA coefficients are computed using training samples and then
        correlation is evaluated on test samples.
        """
        corr_all = {}
        for dataset in DATASETS:
            corr_all[dataset.dataset_name] = []
            rel_ge_data = dataset.train_ge_data
            for k in dataset.k_range:
                rel_exposures = self.train_exposures[dataset.dataset_name][k]
                for rep in np.arange(REPEATS):
                    folds = dataset.cca_folds['train'][rep]
                    folds_in_dataset = len(folds)
                    for fold_i in np.arange(folds_in_dataset):
                        not_fold = np.array(np.setdiff1d(np.arange(folds_in_dataset), fold_i))
                        rel_indices_train = np.concatenate([folds[f] for f in not_fold])
                        rel_indices_val = folds[fold_i]
                        cca = rcca.CCA(reg=1e-4, numCC=1, verbose=False)
                        cca.train([rel_exposures[rel_indices_train], rel_ge_data[rel_indices_train]])
                        corr = self.__compute_correlation(cca, rel_exposures[rel_indices_val],
                                                          rel_ge_data[rel_indices_val])
                        corr_all[dataset.dataset_name].append(corr)
            self.fitness_scores[dataset.dataset_name] = np.round(np.mean(corr_all[dataset.dataset_name]), 4)
        print('Correlation:', self.fitness_scores)

    def assign_kmers_to_categories(self):
        """
        For each dataset, assigns mutation sequences (kmers) to categories. First, performs passes over some 2000 random
        sequences in a dataset to resolve assignment conflicts, until a threshold of ALLOWED_FRACTION_OF_CONFLICTS
        is achieved. Then, performs final assignment for all sequences.
        """
        np.random.seed()
        flag_empty_in_categorization = False
        for category in self.categories:
            if not flag_empty_in_categorization:
                if category == EMPTY_CATEGORY:
                    flag_empty_in_categorization = True
                    continue
            category.compute_scores()
        if not flag_empty_in_categorization:
            self.categories[np.random.choice(len(self.categories), size=1, replace=False)[0]] = EMPTY_CATEGORY
        count_conflicts = np.inf
        count_loops = 0
        n_total_kmers = len(DATASETS[0].all_kmers)
        n_conflicts_allowed = int(n_total_kmers * ALLOWED_FRACTION_OF_CONFLICTS)
        while count_conflicts > n_conflicts_allowed:
            count_loops += 1
            # first pass to reduce conflicts
            empty_category_idx = -1
            for j, category in enumerate(self.categories):
                if category == EMPTY_CATEGORY:
                    empty_category_idx = j
                    break
            for i, kmer in enumerate(np.random.choice(DATASETS[0].all_kmers, size=2000, replace=False)):
                rel_cat_indices = []
                for j, category in enumerate(self.categories):
                    if j == empty_category_idx:
                        continue
                    kmer_fits_category = self.__check_match(kmer, category)
                    if kmer_fits_category:  # potential category <=> no mismatches in category
                        rel_cat_indices.append(j)
                if len(rel_cat_indices) > 1:
                    flag_success, indices_of_best_by_criteria = self.__find_best_match(rel_cat_indices)
                    if not flag_success:
                        np.random.shuffle(indices_of_best_by_criteria)
                        idx_to_del = indices_of_best_by_criteria[1:]
                        self.categories = [category for i, category in enumerate(self.categories) if i not in
                                           idx_to_del]
                        replacement_categories = self.__generate_categories(n_generate=len(idx_to_del))
                        for category in replacement_categories: category.compute_scores()
                        self.categories.extend(replacement_categories)
                        for j, category in enumerate(self.categories):
                            if category == EMPTY_CATEGORY:
                                empty_category_idx = j
                                break
            for dataset in DATASETS:
                # now assign
                assignment = []
                count_conflicts = 0
                empty_category_idx = -1
                for j, category in enumerate(self.categories):
                    if category == EMPTY_CATEGORY:
                        empty_category_idx = j
                        break
                for i, kmer in enumerate(dataset.all_kmers):
                    rel_cat_indices = []
                    for j, category in enumerate(self.categories):
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
                print('number of conflicts in assignment process: %d, loop: %d' % (count_conflicts, count_loops))
                self.assignments[dataset.dataset_name] = np.array(assignment)
        self.updated_nmf = False

    def mutate(self):
        """
        Randomly mutates categories in the categorization, with MUTATION_RATE probability.
        """
        np.random.seed()
        categories_to_mutate = []
        for i in np.arange(len(self.categories)):
            mutation_flag = random.uniform(0, 1) < MUTATION_RATE
            if not mutation_flag: continue
            if self.categories[i] == EMPTY_CATEGORY: continue
            categories_to_mutate.append(i)

        for idx in categories_to_mutate:
            new_cat = self.categories[idx]
            rel_for_mutation_idx = [idx for idx in np.arange(len(new_cat.sequence)) if
                                    idx + new_cat.start_idx not in [SEQ_LENGTH // 2 - 1, SEQ_LENGTH // 2]]
            base_to_mutate_idx = np.random.choice(rel_for_mutation_idx, size=1)[0]
            new_base = np.random.choice([base for base in BASES[1:] if
                                         len(np.intersect1d(BASE_DICT[base],
                                                            BASE_DICT[new_cat.sequence[base_to_mutate_idx]])) != 0
                                         and base != new_cat.sequence[base_to_mutate_idx]])
            final_new_cat = Category(new_cat.start_idx,
                                     new_cat.sequence[:base_to_mutate_idx] + new_base + new_cat.sequence[
                                                                                        base_to_mutate_idx + 1:])
            if final_new_cat not in self.categories:
                self.categories[idx] = final_new_cat

    def evaluate(self):
        """
        Evaluates a categorization: assigns sequences to categories, produces signature and exposure matrices, then
        computes correlation between exposure and gene expression (categorization's fitness).
        """
        self.assign_kmers_to_categories()
        self.__run_nmf()
        self.__compute_fitness()

        return self, self.fitness_scores


class Dataset(object):
    def __init__(self, path_to_wgs_mutation_data=None, path_to_mutation_data=None, path_to_ge_data=None,
                 dataset_name=None, optimal_k=None, k_range=None):
        """
        Initializes a dataset. Defines samples for signature extraction and for canonical correlation analysis,
        splits the CCA data to folds for cross validation, extracts a list of mutation sequences (kmers).
        :param path_to_wgs_mutation_data: path to mutation dataset used for signature learning (string).
        :param path_to_mutation_data: path to mutation dataset used for canonical correlation analysis (string).
        :param path_to_ge_data: path to gene expression dataset used for canonical correlation analysis (string).
        :param dataset_name: string.
        :param optimal_k: optimal number of NMF components, number of dataset mutational signatures;
        retrieved independently using CV2K method (int).
        :param k_range: range for the number of components to be considered for fitness evaluation.
        """
        self.dataset_name = dataset_name
        # read data
        ge_data = pd.read_csv(path_to_ge_data, low_memory=False)
        mutation_data = pd.read_csv(path_to_wgs_mutation_data, low_memory=False)
        wxs_mutation_data = pd.read_csv(path_to_mutation_data, low_memory=False)
        mutation_data = mutation_data.merge(wxs_mutation_data, on='kmer', how='outer').fillna(0).astype(int,
                                                                                                        errors='ignore')
        # split data to sig and cca samples
        samples = copy.deepcopy(mutation_data.columns[1:].to_numpy())
        all_indices = np.arange(len(samples))
        n_cca_samples = len(wxs_mutation_data.columns[1:])
        n_sig_samples = len(samples) - n_cca_samples
        self.sig_sample_indices = all_indices[:n_sig_samples]
        self.cca_sample_indices = all_indices[n_sig_samples:]

        # adjust mutation datasets to kmer size
        if SEQ_LENGTH < 10:
            par = (10 - SEQ_LENGTH) // 2
            temp_catalog = mutation_data.groupby(mutation_data['kmer'].str[par:-par]).sum()
            temp_catalog = temp_catalog.reset_index()
            mutation_data = temp_catalog

        # get list of kmers
        self.all_kmers = copy.deepcopy(mutation_data['kmer'].to_numpy())
        for i, cat in enumerate(self.all_kmers):
            self.all_kmers[i] = cat.replace('[', '').replace(']', '').replace('>', '')
        self.sig_samples = samples[self.sig_sample_indices]
        all_samples = mutation_data.columns[1:]
        ge_samples = ge_data.to_numpy()
        ge_sample_indices = np.where(np.in1d(all_samples, ge_samples))[0]
        self.cca_sample_indices = np.intersect1d(self.cca_sample_indices, ge_sample_indices)

        # adjust ge data
        cca_samples = all_samples[self.cca_sample_indices]
        self.train_ge_data = ge_data.loc[ge_data['icgc_specimen_id'].isin(cca_samples)]
        print(mutation_data.shape, len(self.cca_sample_indices), self.train_ge_data.shape,
              np.array_equal(cca_samples, self.train_ge_data['icgc_specimen_id']))
        self.train_ge_data = self.train_ge_data[
            np.intersect1d(self.train_ge_data.columns, DDR_GENE_SET)].to_numpy()  # only relevant genes
        self.mutation_data = mutation_data

        # split to train set of n_folds folds and test set of n_folds folds
        n_test_samples = len(cca_samples)//2 if len(cca_samples)//2 <= 250 else 250
        self.test_samples_idx = np.random.choice(len(self.cca_sample_indices), size=n_test_samples, replace=False)
        self.test_samples = cca_samples[self.test_samples_idx]
        self.train_samples_idx = np.setdiff1d(np.arange(len(self.cca_sample_indices)), self.test_samples_idx)
        self.train_samples = cca_samples[self.train_samples_idx]
        self.cca_folds = {'train': {}, 'test': {}}
        for rep in np.arange(REPEATS):
            indices_train = np.random.permutation(self.train_samples_idx)
            self.cca_folds['train'][rep] = np.array_split(indices_train, N_FOLDS)
            indices_test = np.random.permutation(self.test_samples_idx)
            self.cca_folds['test'][rep] = np.array_split(indices_test, N_FOLDS)

        # read and determine k data
        self.optimal_k = optimal_k
        self.k_range = k_range
        if k_range is None:
            self.k_range = np.arange(max(optimal_k - 2, 2), optimal_k + 3)


class GeneticAlgorithm(object):
    def __init__(self, population_size=40, ancestor_size=None, n_workers=None,
                 run_name=None):
        """
        Initializes the genetic algorithm.
        :param population_size: number of categorizations in the population throughout the generations (int).
        :param ancestor_size: number of categorizations in the first generation - randomly initialized (int).
        :param n_workers: number of workers (int).
        :param run_name: name of run - output folder (string).
        """
        self.start_time = time.time()
        self.identity_threshold = 0.3
        self.population_size = population_size
        self.ancestor_size = ancestor_size
        if self.ancestor_size is None:
            self.ancestor_size = self.population_size
        self.n_to_keep = 5

        self.current_generation = 0
        self.run_name = run_name
        if not os.path.exists(self.run_name):
            os.makedirs(self.run_name)

        self.n_workers = n_workers
        if self.n_workers is None:
            self.n_workers = self.population_size
        self.population = []
        self.standard_fitness = {}

        self.__ancestor()
        self.train_fitness = []
        self.categorization_by_all_fitness = {}
        self.offspring = []
        self.train_fitness_history = []
        self.test_fitness_history = []

    def __probs(self, values):
        """
        Calculates probabilities.
        :param values: list of ranks.
        :return: probability function (list).
        """
        p = []
        values = np.max(values) + 1 - values
        denom = np.sum(values ** SELECTION_POWER)
        for value in values:
            p.append(value ** SELECTION_POWER / denom)
        return p

    def __ancestor(self):
        """
        Initializes the first generation - ancestor_size categorizations.
        """
        with Pool(self.n_workers) as pool:
            population = [pool.apply_async(Categorization, ()) for i in range(self.ancestor_size)]
            pool.close()
            pool.join()
        self.population = [x.get() for x in population]

    def __selection(self):
        """
        Evaluates categorizations' fitness and produces rankings according to each dataset as well as an average
        overall ranking.
        """
        with Pool(self.n_workers) as pool:
            results = [pool.apply_async(categorization.evaluate, ()) for categorization in self.population]
            pool.close()
            pool.join()
        results = [x.get() for x in results]
        self.population = [x[0] for x in results]
        self.train_fitness = [x[1] for x in results]

        average_ranking = np.zeros((1, len(self.population))).flatten()
        for dataset_name in self.population[0].fitness_scores.keys():
            cur_dataset_scores = [curr_categorization.fitness_scores[dataset_name] for curr_categorization in
                                  self.population]
            self.categorization_by_all_fitness[dataset_name] = np.argsort(cur_dataset_scores)[::-1]
            for rank, idx in enumerate(self.categorization_by_all_fitness[dataset_name]):
                average_ranking[idx] += rank
        self.categorization_by_all_fitness['overall'] = np.argsort(average_ranking)

    def __population_crossover(self, n_offspring):
        """
        Performs crossover between categorizations in the population to produce n_offspring offspring. For each
        offspring, two parent categorizations are picked - each with some probability based on its ranking in a randomly
        chosen ranking system.
        :param n_offspring: number of offspring to produce (int).
        """
        print("Crossing over population to produce %d offspring" % n_offspring)
        parents_idx_lst = []
        parents_lst = []
        for i in range(n_offspring):
            parents_indices = [-1, -1]
            idx = []
            while parents_indices[0] == parents_indices[1]:
                idx1, idx2 = np.random.choice(np.arange(len(self.population)), size=2, replace=True,
                                              p=self.__probs(np.arange(len(self.population))))
                idx = np.sort([idx1, idx2])
                rank1, rank2 = np.random.choice(DATASET_NAMES + ['overall'], size=2, replace=True)
                parents_indices = [self.categorization_by_all_fitness[rank1][idx[0]],
                                   self.categorization_by_all_fitness[rank2][idx[1]]]
            print("Mating parents are ranked:", idx)
            parents = [self.population[parents_indices[0]], self.population[parents_indices[1]]]
            parents_idx_lst.append(parents_indices)
            parents_lst.append(parents)
        with Pool(self.n_workers) as pool:
            offspring = [pool.apply_async(single_crossover, (parents_lst[i],)) for i in range(n_offspring)]
            pool.close()
            pool.join()
        self.offspring = [x.get() for x in offspring]

    def __document(self, best_categorization):
        """
        Documents the current generation (fitness scores, top categorizations, etc.).
        :param best_categorization: the category with the highest fitness score in the population.
        """
        gen_directory = self.run_name + '/gen%d/' % self.current_generation
        if not os.path.exists(gen_directory):
            os.makedirs(gen_directory)
        overall_runtime = str(timedelta(seconds=time.time() - self.start_time))
        if self.current_generation == 1:
            for i, dataset in enumerate(DATASETS):
                np.save(gen_directory + '%s_test_samples.npy' % DATASET_NAMES[i], dataset.test_samples)
        np.save(gen_directory + 'categories_of_best.npy', best_categorization.categories)
        for dataset_name in DATASET_NAMES:
            cat_to_save = self.population[self.categorization_by_all_fitness[dataset_name][0]]
            np.save(gen_directory + 'categories_of_best_%s.npy' % dataset_name, cat_to_save.categories)
        arg_of_junk = \
        [i for i, category in enumerate(best_categorization.categories) if category == EMPTY_CATEGORY][0]
        percent_of_junk = np.count_nonzero(best_categorization.assignments == arg_of_junk) / len(
            best_categorization.assignments)
        self.train_fitness_history.append(best_categorization.fitness_scores)
        self.test_fitness_history.append(best_categorization.validation_scores)
        avg_length = np.mean([len(category.sequence) for category in best_categorization.categories])
        best_catalog = best_categorization.catalogs[DATASETS[0].dataset_name]['train']

        print("::Best categorization data::\n")
        for dataset_name, score in best_categorization.fitness_scores.items():
            print("%s: %.4f" % (dataset_name, score))
        print("Average length of category: %.2f" % avg_length)
        print("Percentage of junk: %.4f" % percent_of_junk)
        print("empty categories: %d" % np.count_nonzero(np.sum(best_catalog, axis=0) == 0))

        print("\nOverall runtime: %s\n" % overall_runtime)

        with open('%s/info.txt' % gen_directory, 'w') as file:
            file.write("::Best categorization data::\n")
            for dataset_name, score in best_categorization.fitness_scores.items():
                file.write("%s: %.4f\n" % (dataset_name, score))
            file.write("Average length of category: %.2f\n" % avg_length)
            file.write("Percentage of junk: %.4f\n" % percent_of_junk)
            file.write("empty categories: %d\n\n" % np.count_nonzero(np.sum(best_catalog, axis=0) == 0))
            file.write("Overall runtime: %s\n" % overall_runtime)

    def one_generation(self):
        """
        Runs a single generation - performs selection, documentation, creation of next generation's population through
        crossover and mutation.
        """
        self.current_generation += 1
        gen_start = time.time()
        print("Calculating fitness and performing selection...")
        self.__selection()
        best_categorization_idx = self.categorization_by_all_fitness['overall'][0]
        best_categorization = self.population[best_categorization_idx]
        print("Done.\n")

        print("Documenting generation %d..." % self.current_generation)
        self.__document(best_categorization)
        print("Done.\n")

        print("Performing crossover and mutation in population...")
        print("Catergorizations in population before crossover:", len(self.population))
        cats_to_keep_idx = [self.categorization_by_all_fitness[dataset][0] for dataset in DATASET_NAMES]
        if best_categorization_idx not in cats_to_keep_idx:
            cats_to_keep_idx.append(best_categorization_idx)
        for idx in self.categorization_by_all_fitness['overall']:
            flag = 0
            if idx in cats_to_keep_idx:
                continue
            for keep_idx in cats_to_keep_idx:
                if len([category for category in self.population[idx].categories if
                        category in self.population[keep_idx].categories]) / N_CATEGORIES > self.identity_threshold:
                    flag = 1
                    break
            if flag == 1:
                continue
            cats_to_keep_idx.append(idx)
            if len(cats_to_keep_idx) >= self.n_to_keep:
                break
        print("kept:", cats_to_keep_idx)
        cats_to_keep = [self.population[i] for i in cats_to_keep_idx]
        n_offspring = self.population_size - len(cats_to_keep_idx)
        self.__population_crossover(n_offspring)
        print("Catergorizations kept:", len(cats_to_keep_idx), "Offspring produced:", len(self.offspring))
        new_population = cats_to_keep + self.offspring
        self.population = new_population
        print("Done.\n")
        print("Catergorizations in population:", len(self.population))
        gen_end = time.time()
        gen_runtime = str(timedelta(seconds=gen_end - gen_start))
        print("Generation Runtime:", gen_runtime)


def single_crossover(parents):
    """
    Gets two parent categorizations from the current generation and performs crossover and/or mutation to produce an
    offspring for the next generation.
    :param parents: two parent categorizations.
    :return: a single categorization - the offspring.
    """
    np.random.seed()
    crossover_flag = random.uniform(0, 1) < Q_CROSSOVER
    if not crossover_flag:  # only mutate fit categorization
        offspring = Categorization(copy.deepcopy(parents[0].categories))
    else:  # perform crossover and then mutate offspring
        take_percentage_weak = np.random.uniform(MUTATION_RATE, 0.4)
        take_percentage_strong = 1.0 - take_percentage_weak
        s_strong_parent, s_weak_parent = len(parents[0].categories), len(parents[1].categories)
        take_from_weak = int(N_CATEGORIES * take_percentage_weak)
        take_from_strong = int(N_CATEGORIES * take_percentage_strong)
        if take_from_strong > s_strong_parent:
            take_from_strong = s_strong_parent
        while take_from_weak + take_from_strong < N_CATEGORIES:
            take_from_weak += 1
        categories_not_in_fit_parent = [category for category in parents[1].categories if
                                        category not in parents[0].categories]
        if len(categories_not_in_fit_parent) >= take_from_weak:
            categories_from_weak_parent = np.random.choice(len(categories_not_in_fit_parent), size=take_from_weak,
                                                           replace=False)
            categories_from_fit_parent = np.random.choice(s_strong_parent, size=take_from_strong, replace=False)
            categories = [category for i, category in enumerate(parents[0].categories) if
                          i in categories_from_fit_parent] + [category for i, category in
                                                              enumerate(categories_not_in_fit_parent) if
                                                              i in categories_from_weak_parent]
            offspring = Categorization(categories)
        else:
            categories_from_fit_parent = np.random.choice(s_strong_parent, size=take_from_strong, replace=False)
            if len(categories_not_in_fit_parent) > 0:
                categories = [category for i, category in enumerate(parents[0].categories) if
                              i in categories_from_fit_parent] + [category for category in categories_not_in_fit_parent]
            else:
                categories = [category for i, category in enumerate(parents[0].categories) if
                              i in categories_from_fit_parent]
            offspring = Categorization(categories)
    # perform mutation and assignment
    offspring.mutate()
    return offspring


if __name__ == "__main__":
    # parse script parameters
    parser = argparse.ArgumentParser(description='categorization_learning')
    # general parameters
    parser.add_argument('--datatype', type=str, default='example', help='datasets separated with ,')
    parser.add_argument('--opt_k', type=str, default='5', help='optimal Ks separated with ,')
    parser.add_argument('--kmer', type=int, default=7, choices=[7], help='7mer sequences')
    # genetic algorithm parameters
    parser.add_argument('--m', type=float, default=.05, choices=[.01, .05, .1], help='mutation rate')
    parser.add_argument('--q', type=float, default=.5, choices=[0, .5, .7, 1], help='crossover rate')
    parser.add_argument('--p', type=int, default=7, choices=[3, 5, 7], help='selection power')
    parser.add_argument('--c', type=int, default=10, choices=[1, 5, 10, 20],
                        help='center of mass of category size dirichlet dist.')
    args = parser.parse_args()

    # global data and parameters
    BASES = ['N', 'A', 'G', 'C', 'T', 'W', 'M', 'K', 'R', 'Y', 'S']
    BASE_DICT = {'A': ['A'], 'G': ['G'], 'C': ['C'], 'T': ['T'], 'R': ['A', 'G'], 'Y': ['C', 'T'], 'M': ['A', 'C'],
                 'W': ['A', 'T'], 'S': ['C', 'G'], 'K': ['G', 'T'], 'N': ['A', 'C', 'G', 'T']}
    SEQ_LENGTH = args.kmer+1
    C_DIRICHLET = args.c
    Q_CROSSOVER = args.q
    MUTATION_RATE = args.m
    SELECTION_POWER = args.p
    N_CATEGORIES = 96
    ALLOWED_FRACTION_OF_CONFLICTS = 0.01
    REPEATS = 3
    N_FOLDS = 10
    POPULATION_SIZE, ANCESTOR_SIZE = 40, 100
    GENERATIONS_TO_RUN = 500

    PATH_TO_INPUT = 'data/'
    DDR_GENE_SET = np.load(PATH_TO_INPUT + 'DDR_genes.npy', allow_pickle=True)
    WGS_OPPORTUNITY_FILE = PATH_TO_INPUT + 'wgs_opportunity_7mer_dictionary.pickle'
    WXS_OPPORTUNITY_FILE = PATH_TO_INPUT + 'wxs_opportunity_7mer_dictionary.pickle'
    with open(WGS_OPPORTUNITY_FILE, 'rb') as handle:
        WGS_OPPORTUNITY = pickle.load(handle)
    with open(WXS_OPPORTUNITY_FILE, 'rb') as handle:
        WXS_OPPORTUNITY = pickle.load(handle)
    KMER_TO_IDX = {kmer: idx for idx, kmer in enumerate(WXS_OPPORTUNITY.keys())}
    EMPTY_CATEGORY = Category(start_idx=0, sequence='N' * SEQ_LENGTH)
    DATASETS = []
    DATASET_NAMES = [dataset.strip() for dataset in args.datatype.split(",")]
    OPT_K = [k.strip() for k in args.opt_k.split(",")]

    for i, dataset_name_glb in enumerate(DATASET_NAMES):
        DATASETS.append(Dataset(path_to_wgs_mutation_data=PATH_TO_INPUT + '%s_wgs_catalog.csv' % dataset_name_glb,
                                path_to_mutation_data=PATH_TO_INPUT + '%s_catalog.csv' % dataset_name_glb,
                                path_to_ge_data=PATH_TO_INPUT + '%s_ge.csv' % dataset_name_glb,
                                dataset_name=dataset_name_glb, optimal_k=int(OPT_K[i]), k_range=None))
    timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')
    run_name = '%s_kmer-%d_mut-%.2f_q-%.2f_c-%d_power-%d_%s' % (args.datatype, SEQ_LENGTH, MUTATION_RATE, Q_CROSSOVER,
                                                                C_DIRICHLET, SELECTION_POWER, timestamp)

    GA = GeneticAlgorithm(population_size=POPULATION_SIZE, ancestor_size=ANCESTOR_SIZE, run_name=run_name)
    while GA.current_generation < GENERATIONS_TO_RUN:
        GA.one_generation()
