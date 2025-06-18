import argparse
import sys
from tokenizer1 import Tokenizer1
from train_ner_model import train_ner_model
from train_tokenizer import train_tokenizer
from test_tokenizer import test


def train_model(train_path, output_path, vocab_size=5000):
    train_tokenizer(train_path, output_path, vocab_size)


def train_ner(tokenizer_path, train_file, dev_file , output_path):
    output_dir = output_path
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 20

    f1 = train_ner_model(
        tokenizer_path,
        train_file,
        dev_file,
        output_dir,
        batch_size,
        learning_rate,
        num_epochs
    )

    return f1

def combine_files_sequential(file1_path, file2_path, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            out.write(f1.read())  # write all of file1

        with open(file2_path, 'r', encoding='utf-8') as f2:
            out.write(f2.read())  # append all of file2


if __name__ == "__main__":
    """
    # combine_files_sequential('data/domain_1_train.txt', 'data/domain_2_train.txt', 'data/domain_test_train.txt')

    # train tokenizer 1
    print("------ Starting Training Tokenizer 1 --------")
    #train_model('data/domain_1_train.txt', 'tokenizers/model1', vocab_size=300)
    print("------ Finished Training Tokenizer 1 --------")

    # train tokenizer 2
    print("------ Starting Training Tokenizer 2 --------")
    train_model('data/domain_2_train.txt', 'tokenizers/model2', vocab_size=1500)
    print("------ Finished Training Tokenizer 2 --------")

    # train tokenizer 3
    print("------ Starting Training Tokenizer Test --------")
    train_model('data/domain_test_train.txt', 'tokenizers/test', vocab_size=750)
    print("------ Finished Training Tokenizer Test --------")

    print("------ Starting Training NER Tokenizer 1 --------")

    f1_tokenizer1 = train_ner(
        'tokenizers/model1/tokenizer.pkl', 'data/ner_data/train_1_binary.tagged',
        'data/ner_data/dev_1_binary.tagged', 'models/model1')
    print("------ Finished Training NER Tokenizer 1 --------")

    print("------ Starting Training NER Tokenizer 2 --------")
    f1_tokenizer2 = train_ner(
        'tokenizers/model1/tokenizer.pkl', 'data/ner_data/train_2_binary.tagged',
        'data/ner_data/dev_2_binary.tagged', 'models/model2')

    print("------ Finished Training NER Tokenizer 2 --------")
"""

    print("------ Testing Tokenizer 1 --------")
    test('tokenizers/model1/tokenizer.pkl',
         'data/domain_1_train.txt',
         'data/domain_1_dev.txt')

    print("------ Testing Tokenizer 2 --------")
    test('tokenizers/model2/tokenizer.pkl',
         'data/domain_2_train.txt',
         'data/domain_2_dev.txt')



