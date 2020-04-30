import os
import pickle
import numpy as np
import gensim
import torch

def collate_data(batch):
    images, lengths, answers, families, idxs = [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family, idx = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)
        idxs.append(idx)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), families, idxs

def collate_data_GQA(batch):
    images, nums_objects, lengths, answers, idxs = [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[2]), batch))
    max_num_objects = max(map(lambda x: x[1], batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_q_len = sorted(batch, key=lambda x: len(x[2]), reverse=True)

    for i, b in enumerate(sort_by_q_len):
        image, _, question, length, answer, idx = b
        images.append(image[:,:,:max_num_objects])
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        idxs.append(idx)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), [], idxs

def load_pretrained_embedings(cfg, embeddings_table, device):
  # load pretrained embeddings from disc
  embeddings_w2v = gensim.models.KeyedVectors.load_word2vec_format(cfg.MAC.TRAINED_EMBD_PATH, binary=True)

  # load vocabulary
  features_path = os.path.join(cfg.DATALOADER.FEATURES_PATH, 'features')
  with open('{}/all_dic.pkl'.format(features_path), 'rb') as f:
    vocabulary = pickle.load(f)['word_dic']

  # overwrite default weights
  embedding_weights = embeddings_table.weight.detach()
  for key in vocabulary:
      if key in embeddings_w2v:
          id = vocabulary[key]
          embedding_weights[id] = torch.from_numpy(embeddings_w2v[key]).to(device)
  embeddings_table.weight = torch.nn.Parameter(embedding_weights)
