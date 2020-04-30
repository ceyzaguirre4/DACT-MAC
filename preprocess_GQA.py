import os
import sys
import json
import pickle

import nltk
import tqdm


# # Utils

def process_question(root, split, variant='all', word_dic=None, answer_dic=None):
    assert variant in ['balanced', 'all']

    if word_dic is None:
        word_dic = {}
    if answer_dic is None:
        answer_dic = {}

    file_queue = []
    question_path = os.path.join(root, 'questions')
    split_path = os.path.join(question_path,
                              '{}_{}_questions'.format(split, variant))
    if os.path.isdir(split_path):
        # data is subdivided into files
        file_queue.extend(map(lambda filename: os.path.join(split_path, filename),
                              os.listdir(split_path)))
    elif os.path.exists(split_path + '.json'):
        file_queue.append(split_path + '.json')
    else:
        raise FileNotFoundError

    result = []
    word_index = len(word_dic) + 1
    answer_index = len(answer_dic)

    for file_path in file_queue:
        with open(file_path) as f:
            data = json.load(f)
        for qid, question in tqdm.tqdm(data.items()):
            words = nltk.word_tokenize(question['question'].lower())
            question_token = []

            for word in words:
                try:
                    question_token.append(word_dic[word])

                except:
                    question_token.append(word_index)
                    word_dic[word] = word_index
                    word_index += 1

            answer = int(qid)      # when testing save question_id
            if 'answer' in question:
                answer_word = question['answer']

                try:
                    answer = answer_dic[answer_word]
                except:
                    answer = answer_index
                    answer_dic[answer_word] = answer_index
                    answer_index += 1

            result.append((
                question['imageId'],
                question_token,
                answer,
                # question['types'],
            ))

    features_path = os.path.join(root, 'features')
    with open('{}/{}_{}.pkl'.format(features_path, variant, split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


# # Tokenize and save features

if __name__ == '__main__':
    ROOT = sys.argv[1]
    try:
        VARIANT = sys.argv[2]
        assert VARIANT in ['all', 'balanced']
    except: VARIANT = 'all'

    word_dic, answer_dic = process_question(ROOT, 'train', VARIANT)
    process_question(ROOT, 'testdev', 'balanced', word_dic, answer_dic)
    process_question(ROOT, 'submission', VARIANT, word_dic, answer_dic)
    process_question(ROOT, 'train', 'balanced', word_dic, answer_dic)

    features_path = os.path.join(ROOT, 'features')
    with open('{}/{}_dic.pkl'.format(features_path, VARIANT), 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
