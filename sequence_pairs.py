import random

import numpy as np
from os.path import join, exists
from os import mkdir
from Bio.SeqIO import FastaIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd


# Object containing the two sequences that make up a pair
class Sample:
    def __init__(self, seq_record_1, seq_record_2, is_ppi):
        self.seq_record_1 = seq_record_1
        self.seq_record_2 = seq_record_2
        self.is_ppi = is_ppi

    def __str__(self):
        return '[ {}, {} ({})]'.format(self.seq_record_1.id, self.seq_record_2.id, self.is_ppi)

    def __eq__(self, other):
        return self.seq_record_1.id == other.seq_record_1.id and self.seq_record_2.id == other.seq_record_2.id \
               or self.seq_record_1.id == other.seq_record_2.id and self.seq_record_2.id == other.seq_record_1.id


# Dictionaries to help with synonymous mutation synthetic data generation
AMINO_ACID_DICT = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['AGA', 'AGG', 'CGT', 'CGC', 'CGA', 'CGG'], 'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'], 'E': ['GAA', 'GAG'], 'Q': ['CAA', 'CAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'], 'H': ['CAT', 'CAC'], 'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'K': ['AAA', 'AAG'], 'M': ['ATG'], 'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], 'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'W': ['TGG'], 'Y': ['TAT', 'TAC'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '!': ['TAA', 'TAG', 'TGA']
}

CODON_DICT = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '!', 'TAG': '!', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '!', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}


# Functions for making CGR "images".
# Recursive method to make an array where each cell contains the k-mer to which it corresponds
def fill_kmer_array(prefix, kmer_array):
    if len(kmer_array) == 2:
        kmer_array[0, 0] = prefix + 'A'
        kmer_array[0, 1] = prefix + 'T'
        kmer_array[1, 0] = prefix + 'C'
        kmer_array[1, 1] = prefix + 'G'
    elif len(kmer_array) > 2:
        n = int(len(kmer_array) / 2)
        fill_kmer_array(prefix + 'A', kmer_array[:n, :n])
        fill_kmer_array(prefix + 'T', kmer_array[:n, n:])
        fill_kmer_array(prefix + 'C', kmer_array[n:, :n])
        fill_kmer_array(prefix + 'G', kmer_array[n:, n:])


# Make a CGR attractor at resolution k from the given sequence
def make_cgr(sequence, k, normalise):
    x = 2 ** k
    # make k-mer array
    kmers = np.array([['X' * k for j in range(x)] for i in range(x)])
    fill_kmer_array('', kmers)
    # build a dictionary of cell coordinates for the possible k-mers
    kmer_addresses = {}
    for i in range(x):
        for j in range(x):
            kmer_addresses[kmers[i, j]] = [i, j]
    # make CGR
    cgr = np.zeros((x, x))
    for i in range(len(sequence) - (k - 1)):
        kmer = sequence[i:i + k]  # find subsequence
        # look up corresponding cell
        if kmer in kmer_addresses:
            ki, kj = kmer_addresses[kmer]
            # increment corresponding cell
            cgr[ki, kj] += 1
    if normalise:
        return cgr / np.max(cgr)
    else:
        return cgr


# Generate a new DNA coding region that translates to the same protein sequence as an existing coding region.
def gen_new_sequence(old_sequence, change_chance=1.0):
    new_sequence = ''
    for i in range(0, len(old_sequence), 3):
        codon = old_sequence[i:i + 3].upper()
        # Make sure there are actually alternatives and draw a random number against the chance of a change...
        if codon in CODON_DICT and (change_chance == 1.0 or np.random.uniform(0.0, 1.0, 1) <= change_chance):
            codon_list = AMINO_ACID_DICT[CODON_DICT[codon]]
            new_codon = np.random.choice(codon_list)
        else:
            # If an unknown character has snuck into the codon, or we don't choose to change it, leave as is...
            new_codon = codon
        new_sequence += new_codon
    return str(new_sequence)


class SampleList:

    def __init__(self, sample_list=None):
        if sample_list is None:
            self.sample_list = []
        else:
            # Make from a python list of Sample objects
            self.sample_list = sample_list

    @classmethod
    # Make from two pandas dataframes
    def make_from_dataframes(cls, positive_df, negative_df, sequence_map, id_col1, id_col2):
        new_sample_list = []
        for row in positive_df.iloc:
            new_sample_list.append(Sample(sequence_map[row[id_col1]], sequence_map[row[id_col2]], True))
        for row in negative_df.iloc:
            new_sample_list.append(Sample(sequence_map[row[id_col1]], sequence_map[row[id_col2]], False))
        return cls(new_sample_list)

    # Compile a set of all unique IDs used in the samples
    def get_all_ids(self):
        id_list = set([])
        for sample in self.sample_list:
            id_list.add(sample.seq_record_1.id)
            id_list.add(sample.seq_record_2.id)
        return id_list

    def __contains__(self, item):
        return self.sample_list.__contains__(item)

    def add_sample(self, sample):
        self.sample_list.append(sample)

    # Create new data sets by splitting the samples according to the given ratios into training, validation and
    # test data.
    # Throws an exception if the ratios don't sum to 1.
    def split(self, rng=None, ratios=[0.6, 0.2, 0.2]):
        if rng is None:
            rng = np.random.default_rng()
        num_all = len(self.sample_list)
        if sum(ratios) != 1:
            raise Exception('Incorrect ratio list', 'Ratios sum to ' + str(sum(ratios)) + ', should sum to 1.0')
        num_training = int(ratios[0] * num_all)
        num_val = int(ratios[1] * num_all)
        shuffled_list = self.sample_list.copy()
        rng.shuffle(shuffled_list)
        return SampleList(shuffled_list[:num_training]), \
               SampleList(shuffled_list[num_training:num_training + num_val]), \
               SampleList(shuffled_list[num_training + num_val:])

    # Create test, train, and validation subsets, convert each subset to CGRs then save the data.
    def split_and_save(self, ks, folder, prefix):
        np.random.seed(1)
        random.seed(1)
        rng=np.random.default_rng(1)
        training, validation, test = self.split(rng)
        for k in ks:
            training_x, training_y = training.to_cgr_data(k, normalise=False)
            validation_x, validation_y = validation.to_cgr_data(k, normalise=False)
            test_x, test_y = test.to_cgr_data(k, normalise=False)
            np.savez(join(folder,f'{prefix}_{k}mers.npz'), training=training,
                     validation=validation, test=test,
                     training_x=training_x, training_y=training_y,
                     validation_x=validation_x, validation_y=validation_y,
                     test_x=test_x, test_y=test_y)

    # Create new SampleList object that contains all samples in this list that contain only sequences with an allowed ID
    def remove_ids(self, allowed_id_list):
        excluded = set(self.get_all_ids()).difference(set(allowed_id_list.keys()))
        filtered_samples = []
        for sample in self.sample_list:
            if not (sample.seq_record_1.id in excluded
                    or sample.seq_record_2.id in excluded):
                filtered_samples.append(sample)
        return SampleList(filtered_samples)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        return self.sample_list[item]

    # Output each sequence pair as a FASTA file within either the positive or negative directory (depending on the
    #   pair's true label. The file is named '{first sequence id}-{second sequence id}.fasta'.
    def write_fasta_files(self, positive_directory, negative_directory):
        if not exists(positive_directory):
            mkdir(positive_directory)
        if not exists(negative_directory):
            mkdir(negative_directory)
        print(len(self.sample_list))
        for sample in self.sample_list:
            if sample.is_ppi:
                directory = positive_directory
            else:
                directory = negative_directory
            filename = join(directory, sample.seq_record_1.id + '-' + sample.seq_record_2.id + '.fasta')
            with open(filename, 'w', newline='') as fastafile:
                writer = FastaIO.FastaWriter(fastafile)
                writer.write_record(sample.seq_record_1)
                writer.write_record(sample.seq_record_2)

    def write_dataframe(self, filename):
        results = []
        for sample in self.sample_list:
            results.append([sample.seq_record_1.id, str(sample.seq_record_1.seq), sample.seq_record_2.id,
                            str(sample.seq_record_2.seq), sample.is_ppi])
        dataset_df = pd.DataFrame(results, columns=['ID_1', 'SEQUENCE_1', 'ID_2', 'SEQUENCE_2', 'IS_PPI'])
        dataset_df.to_csv(filename)

    # Output all sequences into one FASTA file
    def write_single_fasta_file(self, filename):
        with open(filename, 'w', newline='') as fastafile:
            writer = FastaIO.FastaWriter(fastafile)
            for sample in self.sample_list:
                writer.write_record(sample.seq_record_1)
                writer.write_record(sample.seq_record_2)

    # Outputs all pairs in the data as mdCGRs, using k-mers of length k. Normalise each CGR if the normalise parameter
    #  is True. Output in shape [number of pairs] x 2^k x 2^k x 2
    def to_cgr_data(self, k, normalise=True):
        inputs = []
        labels = []
        for sample in self.sample_list:
            cgr_1 = make_cgr(sample.seq_record_1.seq, k, normalise)
            cgr_2 = make_cgr(sample.seq_record_2.seq, k, normalise)
            md_cgr = np.dstack((cgr_1, cgr_2))
            inputs.append(md_cgr)
            if sample.is_ppi:
                labels.append(1)
            else:
                labels.append(0)
        return np.stack(inputs), np.stack(labels)

    def add_all(self, other):
        for sample in other.sample_list:
            self.sample_list.append(sample)

    # Generate new synonymous synthetic pairs (multiple = number of pairs to generate)
    # for every pair in the data.
    def add_synthetic_data(self, multiple):
        new_list = SampleList([])
        for sample in self.sample_list:
            for i in range(multiple):
                new_seq_1 = gen_new_sequence(sample.seq_record_1.seq)
                new_seq_2 = gen_new_sequence(sample.seq_record_2.seq)
                new_sample = Sample(
                    SeqRecord(Seq(new_seq_1), sample.seq_record_1.id + '_' + str(i), name=sample.seq_record_1.name,
                              description=sample.seq_record_1.description),
                    SeqRecord(Seq(new_seq_2), sample.seq_record_2.id + '_' + str(i), name=sample.seq_record_2.name,
                              description=sample.seq_record_2.description),
                    sample.is_ppi)
                new_list.add_sample(new_sample)
        self.add_all(new_list)
