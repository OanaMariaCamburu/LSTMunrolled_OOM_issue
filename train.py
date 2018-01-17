import os
import time
import glob
import sys
import math
import spacy
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import random
from random import randint

import torch
import torch.optim as O
import torch.nn as nn
from torchtext import data

from model import SNLIAutoencoder
from model_LSTMCell import SNLIAutoencoderCell
import my_read_snli
import util 
import streamtologger
torch.cuda.set_device(0)
from nltk.translate.bleu_score import corpus_bleu


print("\n\ncudnn: ", torch.backends.cudnn.version())

args = util.get_args()
util.makedirs(args.save_path)
lr = args.lr_adam
if args.optimizer == "sgd":
    lr = args.lr_sgd
elif args.optimizer == "rmsprop":
    lr = args.lr_rmsprop
args.save_title += "_" + args.optimizer + "_lr" + str(lr) + "_hidden" + str(args.d_hidden) + "_n_layers" + str(args.n_layers)
if args.bidir:
    args.save_title += "_bidir"
if args.sanity:
    args.save_title += "_sanity" + str(args.n_data)
if args.spacy:
    args.save_title += "_spacy"
current_run_dir = args.save_path + "/" + time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + args.save_title
util.makedirs(current_run_dir)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.disable_cudnn:
    torch.backends.cudnn.enabled = False

streamtologger.redirect(target=current_run_dir + '/log.txt')

spacy.load("en")
if args.spacy:
    sentences = data.Field(init_token="<SOS>", eos_token="<EOS>", tokenize='spacy', lower=args.lower, include_lengths=True) 
else:
    sentences = data.Field(init_token="<SOS>", eos_token="<EOS>", lower=args.lower, include_lengths=True)
labels = data.Field(sequential=False)
pairs = data.Field(sequential=False)

train, dev, test = my_read_snli.eSNLI.splits(sentences, labels, pairs)

# TODO: build separate vocab for input sentences and output ones
sentences.build_vocab(train, dev, test)
labels.build_vocab(train)
pairs.build_vocab(train)

if args.word_vectors:
    if args.spacy:
        args.vector_cache = '../../vector_cache/input_vectors_autoenc2_bidir_nlayers_spacy.pt'
    if os.path.isfile(args.vector_cache):
        sentences.vocab.vectors = torch.load(args.vector_cache)
    else:
        sentences.vocab.load_vectors(vectors=args.word_vectors)
        util.makedirs(os.path.dirname(args.vector_cache))
        torch.save(sentences.vocab.vectors, args.vector_cache)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=0)

args.n_vocab = len(sentences.vocab)
print("Args  ", args)

if args.resume_snapshot:
    assert(False)
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(0))
else:
    if args.LSTMCell:
        model = SNLIAutoencoderCell(args)
    else:
        model = SNLIAutoencoder(args)
    if args.word_vectors:
        model.embed.weight.data = sentences.vocab.vectors
        model.cuda(0)

print("Number of trainable paramters: ", util.n_parameters(model))

pad_idx = sentences.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()

if args.optimizer == "sgd":
    opt = O.SGD(model.parameters(), lr = args.lr_sgd)
elif args.optimizer == "adam":
    opt = O.Adam(model.parameters(), lr = args.lr_adam)
else:
    opt = O.RMSprop(model.parameters(), lr = args.lr_rmsprop)

iterations = 0
start = time.time()
best_dev_loss = 10000
best_dev_BLEU = 0
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss Test/Loss     Dev/PPL  Test/PPL '
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:8.6f},{:8.6f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f}'.split(','))
print(header)

norm_grads = []
norm_grads_enc = []
norm_grads_dec = []
norm_grads_embed = []
norm_grads_output = []
norm_grads_projection = []

norm_weights = []
norm_weights_enc = []
norm_weights_dec = []
norm_weights_embed = []
norm_weights_output = []
norm_weights_projection = []

train_losses = []
dev_losses_TF = []

dev_ppls = [] # with TF
dev_bleus = [] # with the output of the previous timestep

cumulative_loss = 0
cumulative_N_tokens = 0
current_train_loss_avg = 0

for epoch in range(args.epochs):
    start_epoch = time.time()
    train_iter.init_epoch()

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        
        batch_sentence1 = batch.sentence1[0]
        batch_sentence2 = batch.sentence2[0]
        lens_sentence1 = batch.sentence1[1]
        lens_sentence2 = batch.sentence2[1]
        
        target_input = batch.sentence1[0][:-1]
        target_output = batch.sentence1[0][1:]

        model.train()
        opt.zero_grad()
        if args.train_forloop:
            answer = model(batch, target_input, "for-output-TF")  # batch is T x bs, answer is T x bs x vocab_sizes
        else:
            answer = model(batch, target_input, "batch")

        # print one example from this epoch
        if batch_idx % args.log_every == 0:
            answer_idx = torch.max(answer, 2)[1]
            print("Example from TRAIN at iteration ", iterations)
            util.print_example(batch, sentences.vocab.itos, labels.vocab.itos, answer_idx)

        N_tokens = util.get_Ntokens_batch(target_output, pad_idx)
        loss = criterion(answer.view(answer.size(0) * answer.size(1), -1), target_output.view(target_output.size(0) * target_output.size(1)))
        cumulative_loss += N_tokens * loss.data[0]
        cumulative_N_tokens += N_tokens
        loss.backward() 
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_clip)
        opt.step()

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval() 
            dev_iter.init_epoch() 

            reference_explanations = []
            predicted_explanations = []
            
            dev_loss_TF = 0
            cumulative_N_tokens_dev = 0
            
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                dev_batch_sentence1 = dev_batch.sentence1[0]
                dev_batch_sentence2 = dev_batch.sentence2[0]

                dev_batch_expl_1_input = dev_batch.sentence1[0][:-1]
                dev_batch_expl_1_output = dev_batch.sentence1[0][1:]
                N_tokens_1 = util.get_Ntokens_batch(dev_batch_expl_1_output, pad_idx)
                cumulative_N_tokens_dev += N_tokens_1

                # With output from previous timestep
                answer = model(dev_batch, dev_batch_expl_1_input, "for-output") 
                if dev_batch_idx == 0:
                    answer_idx = torch.max(answer, 2)[1]
                    print("Example from DEV using the output at each time step by FOR loop at iteration ", iterations)
                    example_idx = util.print_example(dev_batch, sentences.vocab.itos, labels.vocab.itos, answer_idx)
                predicted_explanations = util.append_candidates(sentences.vocab.itos, answer, predicted_explanations)
                reference_explanations = util.append_references(sentences.vocab.itos, dev_batch, reference_explanations)

                # With teacher forcing
                answer_1 = model(dev_batch, dev_batch_expl_1_input, "batch")
                dev_loss_TF += N_tokens_1 * criterion(answer_1.view(answer_1.size(0) * answer_1.size(1), -1), dev_batch_expl_1_output.view(dev_batch_expl_1_output.size(0) * dev_batch_expl_1_output.size(1))).data[0]
                if dev_batch_idx == 0:
                    answer_idx_1 = torch.max(answer_1, 2)[1]
                    print("Example from DEV using the correct input at each timestep by batch at iteration ", iterations)
                    example_idx = util.print_example(dev_batch, sentences.vocab.itos, labels.vocab.itos, answer_idx_1, example_idx=example_idx)

            dev_loss_TF = dev_loss_TF / cumulative_N_tokens_dev
            dev_ppl = math.exp(dev_loss_TF)
            dev_bleu = corpus_bleu(reference_explanations, predicted_explanations)

            # Plot losses
            current_train_loss_avg = cumulative_loss / cumulative_N_tokens
            train_losses.append(current_train_loss_avg)
            cumulative_N_tokens = 0
            cumulative_loss = 0
            dev_losses_TF.append(dev_loss_TF)
            train_line, = plt.semilogy(train_losses, "b-", label="train")
            dev_line_TF, = plt.semilogy(dev_losses_TF, "m-", label="dev")
            plt.legend([train_line, dev_line_TF], ['train loss', 'dev loss TF'])
            plt.savefig(current_run_dir + "/losses.png")
            plt.close()

            # Perplexities
            dev_ppls.append(dev_ppl)
            dev_ppl_line, = plt.plot(dev_ppls, "g-", label="dev")
            plt.legend([dev_ppl_line], ['dev ppl'])
            plt.savefig(current_run_dir + "/perplexities.png")
            plt.close()

            # BLEU scores
            dev_bleus.append(dev_bleu)
            dev_bleu_line, = plt.plot(dev_bleus, "g-", label="dev")
            plt.legend([dev_bleu_line], ['dev BLEU'])
            plt.savefig(current_run_dir + "/BLEU.png")
            plt.close()

            # Plot gradients and weights 
            norm_grads.append(util.norm_grads(model, 2))
            norm_grads_enc.append(util.norm_grads(model.encoder, 2))
            norm_grads_dec.append(util.norm_grads(model.decoder, 2))
            norm_grads_output.append(util.norm_grads(model.output, 2))
            norm_grads_projection.append(util.norm_grads(model.projection, 2))

            model_line, = plt.semilogy(norm_grads, "k-", label="model")
            enc_line, = plt.semilogy(norm_grads_enc, "b-", label="enc")
            dec_line, = plt.semilogy(norm_grads_dec, "r-", label="dec")
            output_line, = plt.semilogy(norm_grads_output, "m-", label="output")
            proj_line, = plt.semilogy(norm_grads_projection, "g-", label="proj")
            plt.legend([model_line, proj_line, enc_line, dec_line, output_line], ['model', 'proj', 'enc', 'dec', 'output'])
            plt.savefig(current_run_dir + "/grads.png")
            plt.close()

            # Plot weights
            norm_weights.append(util.norm_weights(model, 2))
            norm_weights_enc.append(util.norm_weights(model.encoder, 2))
            norm_weights_dec.append(util.norm_weights(model.decoder, 2))
            norm_weights_output.append(util.norm_weights(model.output, 2))
            norm_weights_projection.append(util.norm_weights(model.projection, 2))

            model_line, = plt.plot(norm_weights, "k-", label="model")
            enc_line, = plt.plot(norm_weights_enc, "b-", label="enc")
            dec_line, = plt.plot(norm_weights_dec, "r-", label="dec")
            output_line, = plt.plot(norm_weights_output, "m-", label="output")
            proj_line, = plt.plot(norm_weights_projection, "g-", label="proj")
            plt.legend([model_line, proj_line, enc_line, dec_line, output_line], ['model', 'proj', 'enc', 'dec', 'output'])
            plt.savefig(current_run_dir + "/weights.png")
            plt.close()

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), current_train_loss_avg, dev_loss_TF, dev_ppl, dev_bleu))

            # update best valiation set on loss
            if dev_loss_TF < best_dev_loss:
                best_dev_loss = dev_loss_TF
                snapshot_prefix = os.path.join(current_run_dir, 'best_loss_on_dev')
                snapshot_path = snapshot_prefix + '_devppl_{0:.3f}_devBLEU{1:.3f}__iter_{2}.pt'.format(dev_ppl, dev_bleu, iterations)
                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # update best valiation set on BLEU
            if dev_bleu > best_dev_BLEU:
                best_dev_BLEU = dev_bleu
                snapshot_prefix = os.path.join(current_run_dir, 'best_BLEU_on_dev')
                snapshot_path = snapshot_prefix + '_devppl_{0:.3f}_devBLEU{1:.3f}__iter_{2}.pt'.format(dev_ppl, dev_bleu, iterations)
                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), current_train_loss_avg))
    print("time for epoch" + str(epoch) + "  " + util.pretty_duration(time.time() - start_epoch))
