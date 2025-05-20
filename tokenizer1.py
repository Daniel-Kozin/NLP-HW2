from typing import List

from base_tokenizer import BaseTokenizer


class Tokenizer1(BaseTokenizer):

	def __init__(self, vocab_size):
		super(Tokenizer1, self).__init__()
		self.merges = {}
		self.vocab_size = vocab_size
		self.vocab = {idx: bytes([idx]) for idx in range(256)}
		self.scores = {}

	def train(self, texts: List[str]) -> None:
		num_merges = self.vocab_size - 256
		tokens = self.encode_batch(texts)  # List[List[int]]
		ids = list(tokens)  # copy so we don't destroy the original list

		for i in range(num_merges):
			if i % 100 == 0:
				print(f"Iteration {i}")

			idx = 256 + i
			pair, count_pair = self.get_highest_count(ids)  # Find the best pair
			self.scores[pair] = count_pair  # Remember that pair score
			# Add this pair to the vocab in his original form
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
			self.merges[pair] = idx

			print(f"merging {pair} into a new token {idx}")
			# Merge the new pair into the text
			ids = self.merge(ids, pair, idx)

	def get_highest_count(self, ids):
		count = {}
		for sentence in ids:
			for pair in zip(sentence, sentence[1:]):
				count[pair] = count.get(pair, 0) + 1

		best_pair = max(count, key=count.get)
		return best_pair, count[best_pair]

	def merge(self, ids, pair, idx):
		new_ids = []
		for sentence in ids:
			new_sentence = []
			i = 0
			while i < len(sentence):
				# If we find the pair to merge
				if i < len(sentence) - 1 and sentence[i] == pair[0] and sentence[i + 1] == pair[1]:
					new_sentence.append(idx)
					i += 2  # skip the next token as it's part of the merged pair

				else:
					new_sentence.append(sentence[i])
					i += 1
			new_ids.append(new_sentence)
		return new_ids

	def encode(self, text: str) -> List[int]:
		tokens = text.encode('utf-8')
		tokens = list(map(int, tokens))
		while True:
			max_pair = None
			max_score = 0
			for pair in zip(tokens, tokens[1:]):
				if pair in self.scores:
					if self.scores[pair] > max_score:
						max_score = self.scores[pair]
						max_pair = pair
			if max_pair is None:
				break
			new_token = self.merges[max_pair]
			tokens = self.merge(tokens, max_pair, new_token)
		return tokens


	def decode(self, token_ids: List[int]) -> str:
		tokens = b"".join(self.vocab[idx] for idx in token_ids)
		text = tokens.decode("utf-8", errors="replace")
		return text
