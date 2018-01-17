import os
import torch
import math
from argparse import ArgumentParser
from torch.autograd import Variable
import random
from random import randint
from torch.nn.utils import rnn
import numpy as np


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    
    # datasets
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--dev_file', type=str, default='dev.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--sanity', action='store_true', dest='sanity')
    parser.add_argument('--spacy', action='store_true', dest='spacy')
    parser.add_argument('--bidir', action='store_true', dest='bidir')
    parser.add_argument('--LSTMCell', action='store_true', dest='LSTMCell')  
    parser.add_argument('--disable_cudnn', action='store_true', dest='disable_cudnn')
    parser.add_argument('--train_forloop', action='store_true', dest='train_forloop')
    parser.add_argument('--n_data', type=int, default=10)

    # important hyper
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=300) # word embed
    parser.add_argument('--d_hidden', type=int, default=700)
    parser.add_argument('--n_layers', type=int, default=1)

    # Optim
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_adam', type=float, default=.001)
    parser.add_argument('--lr_sgd', type=float, default=.1)
    parser.add_argument('--lr_rmsprop', type=float, default=.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_clip', type=float, default=0.5)
    parser.add_argument('--dp_ratio', type=float, default=0.2)

    # Printing
    parser.add_argument('--dev_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--save_title', type=str, default='')

    parser.add_argument('--use_batchnorm', action='store_true', dest='use_batchnorm')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '../../vector_cache/input_vectors_autoenc2_bidir_nlayers.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='')

    args = parser.parse_args()
    return args

args = get_args()


def remove_file(file):
    try:
        os.remove(file)
    except Exception as e:
        print("\n\nCouldn't remove " + file + " because ", e)
        pass


def out_file_name(in_file, prefix, file_type):
    split_in_file = in_file.split("/")
    out_name = ""
    for i in range(len(split_in_file) - 1):
        out_name += split_in_file[i] + "/"
    out_name += prefix + split_in_file[-1].split(".")[0] + file_type
    return out_name


def arrange_bidir(h):
    #return torch.stack([torch.cat(pair, dim=1) for pair in h.chunk(2, dim=0)])
    return torch.cat(
    [
        torch.index_select(h, 0, Variable(torch.LongTensor(range(0, h.size(0), 2)).cuda())),
        torch.index_select(h, 0, Variable(torch.LongTensor(range(1, h.size(0), 2)).cuda()))
    ], dim=2)    


def n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


# Takes a batch and the lengths of each sequence and returns the PackSequence and the sorted indices
def create_sequence(batch, lengths, batch_first=False):
    sorted_lengths, sorted_indices = torch.sort(lengths, 0, descending=True)
    sorted_batch = batch[:, sorted_indices, :]
    seq = rnn.pack_padded_sequence(sorted_batch, sorted_lengths.tolist(), batch_first=batch_first)
    return seq, sorted_indices


# Given h_t or c_t of a PackedSequence output RNN, sort them back to original order
def recover_order_hiddens(batch, sorted_indices):
    original_indices = sorted(range(len(sorted_indices)), key=sorted_indices.__getitem__)
    original_batch = batch[:, original_indices, :]
    return original_batch


def get_Ntokens_batch(batch_target, pad_idx):
    batch_size = batch_target.size(1)
    count = 0
    for i in range(batch_size):
        target = batch_target[:, i]
        for t in range(len(target)):
            count += 1
            if target[t].data[0] == pad_idx:
                count -= 1 # because I already added it
                break
    return count


def one_hot_vector(inputs, vec_size):
    one_hot = Variable(torch.zeros(*inputs.size(), vec_size))
    if inputs.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(
            one_hot.dim() - 1,
            inputs.unsqueeze(one_hot.dim() - 1),
            1
    )
    return one_hot


def assert_sizes(t, dims, sizes):
    assert(t.dim() == dims)
    for i in range(dims):
        assert t.size(i) == sizes[i], "in size " + str(i) + " given is " + str(t.size(i)) + " expected " + str(sizes[i])


def _get_sentence_from_indices(dictionary, tensor_indices):
    s = ''
    n = len(tensor_indices)
    for i in range(n):
        if i == 0:
            s = dictionary[tensor_indices.data[i]]
        else:
            s = s + ' ' + dictionary[tensor_indices.data[i]]
    return s  


def _get_sentence_beforeEOS_from_indices(dictionary, tensor_indices):
    s = ''
    for i in range(len(tensor_indices)):
        w = dictionary[tensor_indices[i].data[0]]
        if w != "<SOS>" and w != "<EOS>":
            s += ' ' + w
        if w == "<EOS>":
            break
    #print(s)
    return s  


def write_to_csv(writer, batch, answer, vocab, label_vocab, pairs_vocab):
    bs = batch.sentence1[0].size(1)
    prediction_indices = torch.max(answer, 1)[1]
    for i in range(bs):
        write_row = [pairs_vocab[batch.pairID[i].data[0]], label_vocab[batch.gold_label[i].data[0]]]
        sent1 = batch.sentence1[0][:, i]
        write_row.append(_get_sentence_beforeEOS_from_indices(vocab, sent1))
        sent2 = batch.sentence2[0][:, i]
        write_row.append(_get_sentence_beforeEOS_from_indices(vocab, sent2))
        write_row.append(_get_sentence_beforeEOS_from_indices(vocab, prediction_indices[:, i]))
        writer.writerow(write_row)


def _get_array_words_from_indices_before_EOS(dictionary, tensor_indices):
    s = [] 
    for i in range(len(tensor_indices)):
        w = dictionary[tensor_indices.data[i]]
        if w != "<SOS>" and w != "<EOS>":
            s.append(w)
        if w == "<EOS>":
            break
    return s 


def append_references(vocab, dev_batch, array):
    bs = dev_batch.sentence1[0].size(1)
    for i in range(bs):
        current_references = []
        ref_1 = dev_batch.sentence1[0][1:, i]
        word_array_ref_1 = _get_array_words_from_indices_before_EOS(vocab, ref_1)
        current_references.append(word_array_ref_1)
        array.append(current_references)
    return array


def _get_array_words_from_answer_before_EOS(dictionary, answer):
    s = [] 
    tensor_indices = torch.max(answer, 1)[1]
    for i in range(len(tensor_indices)):
        w = dictionary[tensor_indices.data[i]]
        if w != "<SOS>" and w != "<EOS>":
            s.append(w)
        if w == "<EOS>":
            break
    return s 


def append_candidates(vocab, answer, array):
    bs = answer.size(1)
    for i in range(bs):
        candidate = _get_array_words_from_answer_before_EOS(vocab, answer[:, i])
        array.append(candidate)
    return array


def print_example(batch, vocab, label_vocab, answer,  example_idx=None):
    batch_size = batch.sentence1[0].size(1)
    if example_idx is None:
        example_idx = randint(0, batch_size-1)
    print("sentence1: ", _get_sentence_from_indices(vocab, batch.sentence1[0][:, example_idx]))
    print("DECODED: ", _get_sentence_from_indices(vocab, answer[:, example_idx]))
    print("\n")
    return example_idx


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""
    import os, errno
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def pretty_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def norm_grads(m, k):
    norm = 0.0
    list_params = filter(lambda x: x.grad is not None, m.parameters())
    for submod in list_params:
        norm += math.pow(submod.grad.data.norm(k), k)
    return math.pow(norm, 1./k)


def norm_weights(m, k):
    norm = 0.0
    list_submod = filter(lambda x: x.grad is not None, m.parameters())
    for submod in list_submod:
        norm += math.pow(submod.data.norm(k), k)
    return math.pow(norm, 1./k)


