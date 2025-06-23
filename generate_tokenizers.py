from train_tokenizer import train_tokenizer


def train_model(train_path, output_path, vocab_size=5000):
    train_tokenizer(train_path, output_path, vocab_size)


def combine_files_sequential(file1_path, file2_path, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            out.write(f1.read())  # write all of file1

        with open(file2_path, 'r', encoding='utf-8') as f2:
            out.write(f2.read())  # append all of file2


if __name__ == "__main__":

    combine_files_sequential('data/domain_1_train.txt', 'data/domain_2_train.txt', 'data/domain_test_train.txt')

    # train tokenizer 1
    print("------ Starting Training Tokenizer 1 --------")
    train_model('data/domain_1_train.txt', 'tokenizers/model1', vocab_size=750)
    print("------ Finished Training Tokenizer 1 --------")

    # train tokenizer 2
    print("------ Starting Training Tokenizer 2 --------")
    train_model('data/domain_2_train.txt', 'tokenizers/model2', vocab_size=1000)
    print("------ Finished Training Tokenizer 2 --------")

    # train tokenizer 3
    print("------ Starting Training Tokenizer Test --------")
    train_model('data/domain_test_train.txt', 'tokenizers/test', vocab_size=850)
    print("------ Finished Training Tokenizer Test --------")





