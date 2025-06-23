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


def x():
    """
    # combine_files_sequential('data/domain_1_train.txt', 'data/domain_2_train.txt', 'data/domain_test_train.txt')

    # train tokenizer 1"""
    print("------ Starting Training Tokenizer 1 --------")
    #train_model('data/domain_1_train.txt', 'tokenizers/model1', vocab_size=10000)
    print("------ Finished Training Tokenizer 1 --------")
    """
    # train tokenizer 2
    print("------ Starting Training Tokenizer 2 --------")
    #train_model('data/domain_2_train.txt', 'tokenizers/model2', vocab_size=1000)
    print("------ Finished Training Tokenizer 2 --------")

    # train tokenizer 3
    print("------ Starting Training Tokenizer Test --------")
    # train_model('data/domain_test_train.txt', 'tokenizers/test', vocab_size=1250)
    print("------ Finished Training Tokenizer Test --------")
    """
    print("------ Starting Training NER Tokenizer 1 --------")

    f1_tokenizer1 = train_ner(
        'tokenizers/model1/tokenizer.pkl', 'data/ner_data/train_1_binary.tagged',
        'data/ner_data/dev_1_binary.tagged', 'models/model1')
    print("------ Finished Training NER Tokenizer 1 --------")
    """
    print("------ Starting Training NER Tokenizer 2 --------")
    f1_tokenizer2 = train_ner(
        'tokenizers/model1/tokenizer.pkl', 'data/ner_data/train_2_binary.tagged',
        'data/ner_data/dev_2_binary.tagged', 'models/model2')

    print("------ Finished Training NER Tokenizer 2 --------")


    # If test file is not specified, use the training file
    print("------ Testing Tokenizer 1 --------")
    test('tokenizers/model1/tokenizer.pkl',
         'data/domain_1_train.txt',
         'data/domain_1_dev.txt')""""""

    print("------ Testing Tokenizer 2 --------")"""
    test('tokenizers/model1/tokenizer.pkl',
         'data/domain_1_train.txt',
         'data/domain_1_dev.txt')

def experiment():
    vocab_sizes = [300, 500, 750, 1000, 1250, 1500]
    all_results = {}

    def run_experiments(model_name, domain_train_path, tokenizer_base_path,
                        ner_train_path, ner_dev_path, ner_model_path):
        results = []

        for vocab_size in vocab_sizes:
            print(f"\n------ [{model_name}] Training tokenizer with vocab size {vocab_size} ------")
            train_model(domain_train_path, tokenizer_base_path, vocab_size=vocab_size)

            tokenizer_path = f"{tokenizer_base_path}/tokenizer.pkl"
            print(f"------ [{model_name}] Training NER model for vocab size {vocab_size} ------")
            f1 = train_ner(tokenizer_path, ner_train_path, ner_dev_path, ner_model_path)
            print(f"[{model_name}] Vocab size: {vocab_size} -> F1: {f1:.4f}")
            results.append((vocab_size, f1))

        all_results[model_name] = results

        # Top 5
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"\n==== Top 5 F1 scores for {model_name} ====")
        for vocab_size, f1 in results[:5]:
            print(f"Vocab size: {vocab_size} -> F1: {f1:.4f}")
        print("======================================\n")

    run_experiments(
        model_name="Model 1",
        domain_train_path='data/domain_1_train.txt',
        tokenizer_base_path='tokenizers/model1',
        ner_train_path='data/ner_data/train_1_binary.tagged',
        ner_dev_path='data/ner_data/dev_1_binary.tagged',
        ner_model_path='models/model1'
    )

    run_experiments(
        model_name="Model 2",
        domain_train_path='data/domain_2_train.txt',
        tokenizer_base_path='tokenizers/model2',
        ner_train_path='data/ner_data/train_2_binary.tagged',
        ner_dev_path='data/ner_data/dev_2_binary.tagged',
        ner_model_path='models/model2'
    )

    # Print all results
    print("\n\n========== All F1 Scores ==========")
    for model_name, results in all_results.items():
        print(f"\n--- {model_name} ---")
        for vocab_size, f1 in results:
            print(f"Vocab size: {vocab_size} -> F1: {f1:.4f}")
    print("===================================")

def count_phrase_in_file(filepath: str, phrase: str) -> int:
    count = 0
    phrase_len = len(phrase)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Count overlapping occurrences of phrase in line
            start = 0
            while True:
                idx = line.find(phrase, start)
                if idx == -1:
                    break
                count += 1
                start = idx + 1  # move forward by 1 to allow overlapping matches
    return count


if __name__ == "__main__":
    import pickle

    with open("tokenizers/test/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    #tokenizer.show_bi_gram()

    filename = "data/domain_test_train.txt"
    target_words = ["of the", 'in the', 'for the', 'going to', "for the", "on the", "to the", "have to", "have a", "at the"]

    print("The count of each phrase in the training set (sorted by count):")

    # Count and store in list
    counts = [(word, count_phrase_in_file(filename, word)) for word in target_words]

    # Sort descending by count
    counts.sort(key=lambda x: x[1], reverse=True)

    # Print
    for word, count in counts:
        print(f'"{word}" => {count}')