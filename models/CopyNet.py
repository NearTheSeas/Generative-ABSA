import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F
from data_utils import Language
from utils import to_one_hot, DecoderBase,seq_to_string, tokens_to_seq
from spacy.lang.en import English



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, iput, hidden, lengths):
        # iput batch must be sorted by sequence length
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 3)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(iput)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden


class CopyNetDecoder(DecoderBase):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_length):
        super(CopyNetDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length
        self.embedding = nn.Embedding(len(self.lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru = nn.GRU(2 * self.hidden_size + self.embedding.embedding_dim, self.hidden_size, batch_first=True)  # input = (context + selective read size + embedding)
        self.out = nn.Linear(self.hidden_size, len(self.lang.tok_to_idx))

    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]
        seq_length = encoder_outputs.data.shape[1]

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings + seq_length)))
        sampled_idx = Variable(torch.ones((batch_size, 1)).long())
        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()

        sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding

        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        if keep_prob < 1.0:
            dropout_mask = (Variable(torch.rand(batch_size, 1, 2 * self.hidden_size + self.embedding.embedding_dim)) < keep_prob).float() / keep_prob
        else:
            dropout_mask = None

        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        one_hot_input_seq = to_one_hot(inputs, len(self.lang.tok_to_idx) + seq_length)
        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_input_seq = one_hot_input_seq.cuda()

        for step_idx in range(1, self.max_length):

            if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq, dropout_mask=None):

        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = len(self.lang.tok_to_idx)

        # Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)

        # Call the RNN
        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]

        # Copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
        missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # Generate mechanism
        gen_scores = self.out(output.squeeze(1))  # [b, vocab_size]
        gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too

        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        gen_probs = probs[:, :vocab_size]

        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        copy_probs = probs[:, vocab_size:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10**-10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read


class AttentionDecoder(DecoderBase):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_length):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length

        self.embedding = nn.Embedding(len(lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.embedding_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, len(lang.tok_to_idx))

    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))  # overwrite the encoder hidden state with zeros
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings)))
        sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding
        sos_idx = Variable(torch.ones((batch_size, 1)).long())

        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sos_idx = sos_idx.cuda()

        decoder_outputs = [sos_output]
        sampled_idxs = [sos_idx]

        iput = sos_idx

        dropout_mask = torch.rand(batch_size, 1, self.hidden_size + self.embedding.embedding_dim)
        dropout_mask = dropout_mask <= keep_prob
        dropout_mask = Variable(dropout_mask).float() / keep_prob

        for step_idx in range(1, self.max_length):

            if targets is not None and teacher_forcing > 0.0:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) <= teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                iput = iput.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            output, hidden = self.step(iput, hidden, encoder_outputs, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            _, topi = decoder_outputs[-1].topk(1)
            iput = topi.view(batch_size, 1)
            sampled_idxs.append(iput)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, dropout_mask=None):

        batch_size = prev_idx.shape[0]
        vocab_size = len(self.lang.tok_to_idx)

        # encoder_output * W * decoder_hidden for each encoder_output
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        scores = torch.bmm(encoder_outputs, transformed_hidden).squeeze(2)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(attn_weights, encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)

        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding

        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)

        output = self.out(output.squeeze(1))  # linear transformation to output size
        output = F.log_softmax(output, dim=1)  # log softmax non-linearity to convert to log probabilities

        return output, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, embedding_size, decoder_type):
        super(EncoderDecoder, self).__init__()

        self.lang = lang

        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),
                                  hidden_size,
                                  embedding_size)
        self.decoder_type = decoder_type
        decoder_hidden_size = 2 * self.encoder.hidden_size
        if self.decoder_type == 'attn':
            self.decoder = AttentionDecoder(decoder_hidden_size,
                                            embedding_size,
                                            lang,
                                            max_length)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(decoder_hidden_size,
                                          embedding_size,
                                          lang,
                                          max_length)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

    def forward(self, inputs, lengths, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs

    def get_response(self, input_string):
        use_extended_vocab = isinstance(self.decoder, CopyNetDecoder)

        if not hasattr(self, 'parser_'):
            self.parser_ = English()

        idx_to_tok = self.lang.idx_to_tok
        tok_to_idx = self.lang.tok_to_idx

        input_tokens = self.parser_(' '.join(input_string.split()))
        input_tokens = ['<SOS>'] + [token.orth_.lower() for token in input_tokens] + ['<EOS>']
        input_seq = tokens_to_seq(input_tokens, tok_to_idx, len(input_tokens), use_extended_vocab)
        input_variable = Variable(input_seq).view(1, -1)

        if next(self.parameters()).is_cuda:
            input_variable = input_variable.cuda()

        outputs, idxs = self.forward(input_variable, [len(input_seq)])
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)

        return output_string

    def interactive(self, unsmear):
        while True:
            input_string = input("\nType a message to Amy:\n")
            output_string = self.get_response(input_string)

            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')

            print('\nAmy:\n', output_string)
