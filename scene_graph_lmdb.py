import os
import argparse
import numpy as np
import lmdb
import json
import pickle
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='trainval_test', help='trainval or test')
    parser.add_argument('--obj_ldmb_folder', type=str, help='folder where obj lmdb saved')
    parser.add_argument('--sg_folder', type=str, help='folder where sg saved')
    parser.add_argument('--word_emb_folder', type=str, help='folder saving word dict and emb matrix')
    parser.add_argument('--save_dir', type=str, help='dir to save the lmdb result')
    return parser.parse_args()


class WordEmbedding:
    def __init__(self, embedding_method, word_emb_folder):
        glove_dictionary_file = os.path.join(word_emb_folder, 'glove_dictionary.json')
        glove_word_matrix_file = os.path.join(word_emb_folder, 'glove6b_init_300d.npy')
        fasttext_dictionary_file = os.path.join(word_emb_folder, 'fasttext_dictionary.json')
        fasttext_word_matrix_file = os.path.join(word_emb_folder, 'fasttext_init_300d.npy')
        if embedding_method.lower() == 'glove':
            dictionary_file = glove_dictionary_file
            word_matrix_file = glove_word_matrix_file
        elif embedding_method.lower() == 'fasttext':
            dictionary_file = fasttext_dictionary_file
            word_matrix_file = fasttext_word_matrix_file
        else:
            raise ValueError('{} embedding method is allowed'.format(embedding_method))
        with open(dictionary_file, 'r') as f:
            self.word_to_idx = json.load(f)['word_to_ix']
        self.index_to_vector = np.load(word_matrix_file)

    def __call__(self, token):
        try:
            index = self.word_to_idx[token]
        except KeyError:
            index = self.word_to_idx['unknown']
        vector = self.index_to_vector[index]
        return vector


class SceneGraphEmbedding:
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding

    def __call__(self, image_sg):
        n_objects = len(image_sg)
        image_sg_embedding = np.zeros((n_objects, 900), dtype='float32')

        for obj_idx, obj in enumerate(image_sg.values()):

            # name embedding
            name_embedding = self.word_embedding(obj['name'])

            # attribute embedding
            if len(obj['attributes']):
                attr_embedding = np.zeros((len(obj['attributes']), 300), dtype='float32')
                for attr_idx, attr_name in enumerate(obj['attributes']):
                    attr_embedding[attr_idx] = self.word_embedding(attr_name)
                attr_embedding = np.mean(attr_embedding, 0)
            else:
                attr_embedding = np.zeros((1, 300), dtype='float32')

            # relation embedding
            rel_embedding = np.zeros((len(obj['relations']), 300), dtype='float32')
            for rel_index, rel_entity in enumerate(obj['relations']):
                rel_name = rel_entity['name']
                words = rel_name.split()
                word_embs = np.zeros((len(words), 300), dtype='float32')
                for idx, word in enumerate(words):
                    word_embs[idx] = self.word_embedding(word)
                rel_name_emb = np.mean(word_embs, axis=0)
                subject_emb = self.word_embedding(image_sg[str(rel_entity['object'])]['name'])
                rel_embedding[rel_index] = (rel_name_emb + subject_emb) / 2
            rel_embedding = np.mean(rel_embedding, axis=0)

            image_sg_embedding[obj_idx] = np.concatenate((name_embedding, attr_embedding, rel_embedding), axis=None)

        return image_sg_embedding


def main():
    args = get_args()
    word_embed = WordEmbedding('fasttext', args.word_emb_folder)
    scene_graph_emb = SceneGraphEmbedding(word_embed)

    tiers = args.tiers.split('_')

    if 'trainval' in tiers:
        print('#### Loading train and val scene graph json file...')
        with open(os.path.join(args.sg_folder, 'train_sg.json')) as f:
            train_sg = json.load(f)
        with open(os.path.join(args.sg_folder, 'val_sg.json')) as f:
            val_sg = json.load(f)
        print('#### Successfully loaded')

        # train_val sg lmdb generation
        # get image ids
        train_val_obj_ldmb = os.path.join(args.obj_ldmb_folder, 'tvqa_trainval_obj.lmdb')
        env = lmdb.open(train_val_obj_ldmb,
                        max_readers=1,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        )
        with env.begin(write=False) as txn:
            _image_ids = pickle.loads(txn.get("keys".encode()))

        # calculate memory usage
        sample_id = _image_ids[0].decode()
        sample_sg = train_sg[sample_id]['objects']
        sample_sg_embeddings = scene_graph_emb(sample_sg)
        sg_memory_usage = sample_sg_embeddings.size * sample_sg_embeddings.itemsize
        n_train_val_sg = len(train_sg) + len(val_sg)
        memory_usage = n_train_val_sg * sg_memory_usage

        # create lmdb file
        env = lmdb.open(os.path.join(args.save_dir, 'tvqa_trainval_sg.lmdb'), map_size=memory_usage)
        with env.begin(write=True) as txn:
            for _image_id in tqdm(_image_ids,
                                  unit='image',
                                  desc='train_val scene graph lmdb generation'):
                image_id = _image_id.decode()
                try:
                    image_sg = train_sg[image_id]['objects']
                except KeyError:
                    image_sg = val_sg[image_id]['objects']
                image_sg_embeddings = scene_graph_emb(image_sg)
                txn.put(key=_image_id, value=pickle.dumps(image_sg_embeddings))

    if 'test' in tiers:
        # test sg lmdb generation
        print('#### Loading test scene graph json file...')
        with open(os.path.join(args.sg_folder, 'test_sg.json')) as f:
            test_sg = json.load(f)
        print('#### Successfully loaded')

        # get image ids
        test_obj_ldmb = os.path.join(args.obj_ldmb_folder, 'tvqa_test_obj.lmdb')
        env = lmdb.open(test_obj_ldmb,
                        max_readers=1,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        )
        with env.begin(write=False) as txn:
            _image_ids = pickle.loads(txn.get("keys".encode()))

        # calculate memory usage
        sample_id = _image_ids[0].decode()
        sample_sg = test_sg[sample_id]['objects']
        sample_sg_embeddings = scene_graph_emb(sample_sg)
        sg_memory_usage = sample_sg_embeddings.size * sample_sg_embeddings.itemsize
        n_test_sg = len(test_sg)
        memory_usage = n_test_sg * sg_memory_usage

        # create lmdb file
        env = lmdb.open(os.path.join(args.save_dir, 'tvqa_test_sg.lmdb'), map_size=memory_usage)
        with env.begin(write=True) as txn:
            for _image_id in tqdm(_image_ids,
                                  unit='image',
                                  desc='test scene graph lmdb generation'):
                image_id = _image_id.decode()
                image_sg = test_sg[image_id]['objects']
                image_sg_embeddings = scene_graph_emb(image_sg)
                txn.put(key=_image_id, value=pickle.dumps(image_sg_embeddings))


if __name__ == '__main__':
    main()
