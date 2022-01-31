import os
import subprocess
import numpy as np
from .sequence import Sequence

def parse_blast_results(results):
    """
    Parses Blast results
    Args:
        results: Decoded output from blastp
    Returns:
        a dictonary where key is qseqid, value is score evalue pident values
    """
    parsed = {}
    for line in results.split(os.linesep):
        parts = line.split(",")
        parsed[parts[0]] = parts[1:]

    return parsed

def get_local_blast_results(data_dir, db_path, fasta):
    query_path = os.path.join(data_dir, "fasta.fasta")
    with open(query_path, "w+") as f:
        f.write(fasta)

    # TODO: Enzyme class
    blastp = subprocess.Popen(
        ['blastp', '-db', db_path, "-max_target_seqs", "1", "-outfmt", "10 qseqid score evalue pident",
         "-matrix", "BLOSUM45", "-query", query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    results, err = blastp.communicate()
    return parse_blast_results(results.decode()), err.decode()

def get_protein_sequences(sequences, labels=None, d_scores=None):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)

    Returns:
      array of Sequence objects
    """
    seqs = []
    for index, seq in enumerate(sequences):
        label = None if labels is None else labels[index]
        d_score = None if d_scores is None else d_scores[index]
        seqs.append(Sequence(index, seq, label=label, d_score=d_score))
    return seqs

def sequences_to_fasta(sequences, id_to_enzyme_class, escape=True, strip_zeros=False):
    """

    Args:
      sequences: a list of Sequences object
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      escape: a flag to determine if special characters needs to be escape. Applicable for text in tersorboard
      strip_zeros: a flag that determines whether zeros are removed from sequences
    Returns:
      string with sequences and additional information that mimics fasta format

    """
    return os.linesep.join([seq.get_seq_in_fasta(id_to_enzyme_class, escape, strip_zeros) for seq in sequences])

def update_sequences_with_blast_results(parsed_results, sequences):
    """
    Parses results from blasp into separate arrays
    Args:
        parsed_results: Parsed results from blastp
        sequences: sequences used in blastp

    Returns:
        Returns lists of sequences, e.values, similarities scores and identities
    """
    similarities, evalues, identity = [], [], []
    for sequence in sequences:
        if str(sequence.id) in parsed_results:
            result = parsed_results[str(sequence.id)]
            sequence.similarity = float(result[0])
            sequence.evalue = float(result[1])
            sequence.identity = float(result[2])
            similarities.append(sequence.similarity)
            evalues.append(sequence.evalue)
            identity.append(sequence.identity)
    return sequences, evalues, similarities, identity

def get_stats( batch_size, similarities, name, f):
        avg = np.array(similarities).sum() / batch_size
        best_value = f(similarities)
        return avg, best_value