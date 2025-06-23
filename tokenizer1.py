import heapq
from typing import List
from collections import defaultdict
from base_tokenizer import BaseTokenizer
from heapq import heappush, heappop


N = 4


class Tokenizer1(BaseTokenizer):
    def __init__(self, vocab_size):
        """
        Initialize the Tokenizer with vocabulary size and setup initial mappings.
        """
        super(Tokenizer1, self).__init__()
        self.space_token = ' '
        self.merges = {}
        self.vocab_size = vocab_size

        # idx -> pair
        # adds to the 4 special tokens that already exist
        self.id_to_token.update({idx: bytes([idx - N]) for idx in range(4, 256 + N)})
        # token -> idx
        # adds to the 4 special tokens that already exist
        self.token_to_id.update({bytes([idx - N]): idx for idx in range(4, 256 + N)})
        # pair -> amount of appearances
        self.scores = {}
        self.global_count = None

        # for each pair holds the idx of sentences that it is in
        self.pair_to_sen = {}

        # holds all the pairs we've merged
        self.pair_in_vocab = set()


    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer using byte pair encoding (BPE) on a list of texts.
        """
        num_merges = self.vocab_size - 256 - N
        tokens = self.encode_batch(texts)
        ids = list(tokens)  # copy so we don't destroy the original list
        self.global_count = self.get_stats(ids)  # getting initial counts

        # Initializing which pair appears in which sentences
        for i, sen in enumerate(ids):
            for pair in zip(sen, sen[1:]):
                if pair not in self.pair_to_sen:
                    self.pair_to_sen[pair] = set()
                self.pair_to_sen[pair].add(i)

        for i in range(num_merges):
            pair = max(self.global_count, key=self.global_count.get)
            self.scores[pair] = self.global_count[pair]
            idx = 256 + N + i

            token = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx
            self.pair_in_vocab.add((pair[0], pair[1]))

            print(f"merging {pair} ({self.id_to_token[pair[0]]}, {self.id_to_token[pair[1]]})"
                  f" into a new token {idx}")

            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        self.global_count = None
        self.pair_to_sen = None
        """
        # ---- Manual addition of "I am" token ----
        from_bytes = self.shift_encode("I am", N)
        if from_bytes in self.token_to_id:
            print(f'"but I" already in vocabulary')
        else:
            new_token_id = max(self.id_to_token) + 1
            self.token_to_id[from_bytes] = new_token_id
            self.id_to_token[new_token_id] = from_bytes
            self.vocab_size += 1
            print(f'Added manual token "but I" with ID {new_token_id}')
        """
    def get_stats(self, ids):
        """
        Count frequency of all adjacent token pairs in the tokenized sequences.
        """
        counts = defaultdict(int)
        for seq in ids:  # for each sequence of token IDs
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])  # use a tuple, not list
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        Merge a given pair into a new token across all sequences.
        """
        self.global_count[pair] = 0
        for i in self.pair_to_sen[pair]:
            new_ids = []
            j = 0
            while j < len(ids[i]) - 1:
                if ids[i][j] == pair[0] and ids[i][j + 1] == pair[1]:
                    # updating the global count
                    # updating which pairs are in which sentences
                    if j != 0:
                        self.global_count[(ids[i][j - 1], pair[0])] -= 1
                        self.global_count[(ids[i][j - 1], idx)] += 1

                        if (ids[i][j - 1], idx) not in self.pair_to_sen:
                            self.pair_to_sen[(ids[i][j - 1], idx)] = set()
                        self.pair_to_sen[(ids[i][j - 1], idx)].add(i)
                    if j != len(ids[i]) - 2:
                        self.global_count[(pair[1], ids[i][j + 2])] -= 1
                        self.global_count[(idx, ids[i][j + 2])] += 1

                        if (idx, ids[i][j + 2]) not in self.pair_to_sen:
                            self.pair_to_sen[(idx, ids[i][j + 2])] = set()
                        self.pair_to_sen[(idx, ids[i][j + 2])].add(i)
                    new_ids.append(idx)
                    j += 2
                else:
                    new_ids.append(ids[i][j])
                    j += 1
            # checking for the last token
            if j == len(ids[i]) - 1:
                new_ids.append(ids[i][j])

            ids[i] = new_ids
        self.pair_to_sen[pair] = None
        return ids

    def merge_for_encode(self, sen, pair, idx):
        """
        Merge a pair in a single sequence during encoding.
        """
        new_sen = []
        j = 0
        while j < len(sen) - 1:
            if sen[j] == pair[0] and sen[j + 1] == pair[1]:
                new_sen.append(idx)
                j += 2
            else:
                new_sen.append(sen[j])
                j += 1
        # checking for the last token
        if j == len(sen) - 1:
            new_sen.append(sen[j])
        return new_sen

    def shift_encode(self, text, shift):
        """
        Encode text to bytes and apply a byte shift.
        """
        # Encode the text to bytes
        encoded_bytes = text.encode('utf-8')
        # Shift each byte and wrap around at 256 (since bytes are 0–255)
        shifted_bytes = bytes((b + shift) % 256 for b in encoded_bytes)
        return shifted_bytes

    def encode(self, text: str) -> List[int]:
        tokens = list(map(int, self.shift_encode(text, N)))
        if len(tokens) < 2:
            return tokens

        class Node:
            def __init__(self, val):
                self.val = val
                self.prev = None
                self.next = None

        # Build double linked list
        nodes = [Node(tok) for tok in tokens]
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
            nodes[i + 1].prev = nodes[i]

        head = nodes[0]

        # Build heap and position map
        heap = []
        pair_positions = defaultdict(list)

        def add_pair(node):
            if node and node.next:
                pair = (node.val, node.next.val)
                if pair in self.merges:
                    score = self.scores[pair]
                    heappush(heap, (-score, id(node), node))
                    pair_positions[pair].append(node)

        for node in nodes:
            add_pair(node)

        while heap:
            _, _, node = heappop(heap)

            if not node or not node.next:
                continue

            pair = (node.val, node.next.val)
            if pair not in self.merges:
                continue

            # Merge node and node.next
            new_token = self.merges[pair]
            next_node = node.next
            node.val = new_token
            node.next = next_node.next
            if next_node.next:
                next_node.next.prev = node

            # Add new pairs
            add_pair(node.prev)
            add_pair(node)

        # Collect final result
        result = []
        node = head
        while node:
            result.append(node.val)
            node = node.next

        return result

    """
    def encode(self, text: str) -> List[int]:
        tokens = list(map(int, self.shift_encode(text, N)))
        if len(tokens) < 2:
            return tokens

        while True:
            heap = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    heapq.heappush(heap, (-self.scores[pair], i, pair))

            if not heap:
                break

            _, i, pair = heapq.heappop(heap)

            # Confirm pair is still valid
            if i >= len(tokens) - 1 or (tokens[i], tokens[i + 1]) != pair:
                continue

            # Merge the pair
            tokens[i] = self.merges[pair]
            del tokens[i + 1]

        return tokens
    """

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a UTF-8 string.
        """
        tokens = b"".join(self.id_to_token[idx] for idx in token_ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            The number of tokens in the vocabulary.
        """
        return self.vocab_size

    def show_bi_gram(self):
        """
        Print bi-gram tokens from the vocabulary that contain a space followed by a word.
        """
        print("All the bi-grams in the vocabulary that contain a space and have words after it:")
        for (a, b), idx in self.merges.items():
            token = self.id_to_token[idx]
            try:
                decoded = token.decode('utf-8')
            except UnicodeDecodeError:
                continue

            if ' ' in decoded:
                # Split on space
                parts = decoded.split(' ')
                # Check if there is at least one non-empty string after the first space
                # Example: "Hello " -> parts = ['Hello', ''], exclude
                #          "Hello m" -> parts = ['Hello', 'm'], include
                if len(parts) > 1 and any(part.strip() for part in parts[1:]):
                    print(f"{idx}: '{decoded}' (from {a}, {b})")

    def show_bi_gram1(self):
        """
        Print bi-gram tokens from the vocabulary that contain a space followed by a word.
        """
        print("All the bi-grams in the vocabulary that contain a space and have words after it:")
        for (a, b), idx in self.merges.items():
            token = self.id_to_token[idx]
            try:
                decoded = token.decode('utf-8')
            except UnicodeDecodeError:
                continue

            if ' ' in decoded:
                # Split on space
                parts = decoded.split(' ')
                # Check if there is at least one non-empty string after the first space
                # Example: "Hello " -> parts = ['Hello', ''], exclude
                #          "Hello m" -> parts = ['Hello', 'm'], include
                if len(parts) > 1 and any(part.strip() for part in parts[1:]):
                    print(f'{decoded}')