from typing import List
from collections import Counter
import numpy as np
from base_tokenizer import BaseTokenizer


class Tokenizer1(BaseTokenizer):

	def __init__(self, vocab_size):
		super(Tokenizer1, self).__init__()
		self.merges = {}
		self.vocab_size = vocab_size

		# idx -> pair
		self.vocab = {idx: bytes([idx]) for idx in range(256)}
		# token -> idx
		self.token_to_id = {bytes([idx]): idx for idx in range(256)}
		# pair -> amount of appearances
		self.scores = {}
		self.global_count = Counter()

		# for each token holds the idx of sentences that the word is in them
		self.vocab_to_sen = {}
		for i in range(256):
			self.vocab_to_sen[i] = set()

		# holds all the tokens we merged
		self.pair_in_vocab = set()

	def train(self, texts: List[str]) -> None:
		num_merges = self.vocab_size - 256
		tokens = self.encode_batch(texts)  # List[List[int]]
		ids = list(tokens)  # copy so we don't destroy the original list

		sen_changed = [False] * len(ids)
		pair = None

		for i, sentence in enumerate(tokens):
			for token in sentence:
				self.vocab_to_sen[token].add(i)

		for i in range(num_merges):
			if i % 50 == 0:
				self.global_count = Counter({k: v for k, v in self.global_count.items() if v > 0})

			idx = 256 + i
			# Find the best pair
			pair, count_pair = self.get_highest_count(ids, sen_changed, idx - 1, pair)
			# Remember that pair score
			self.scores[pair] = count_pair

			# Add this pair to the vocab in his original form
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
			self.merges[pair] = idx
			self.token_to_id[pair[0] + pair[1]] = idx
			self.vocab_to_sen[idx] = set()

			print(f"merging {pair} ({self.vocab[pair[0]]}, {self.vocab[pair[1]]})"
				  f" into a new token {idx} ({self.vocab[idx]})")

			# get all the sentences were pair[0] and pair[1] apeared
			sen_to_merge = self.vocab_to_sen[pair[0]].intersection(self.vocab_to_sen[pair[1]])
			sen_changed = [False] * len(ids)

			for j in sen_to_merge:
				# Merge the new pair into the text
				new_sen, changed = self.merge(ids[j], pair, idx, j)
				sen_changed[j] = changed
				ids[j] = new_sen

		for pair in self.scores.keys():
			self.pair_in_vocab.add(pair)

	def get_highest_count(self, ids, changed, new_token, pair):
		if any(changed):
			self.global_count[pair] = 0
			for i, sentence in enumerate(ids):
				if not changed[i]:
					continue

				for prev, curr in zip(sentence, sentence[1:]):
					if prev == new_token or curr == new_token:
						self.global_count[(prev, curr)] += 1
						if prev == new_token and curr == new_token:
							continue
						elif prev == new_token:
							self.global_count[(pair[1], curr)] -= 1
						else:
							self.global_count[(prev, pair[0])] -= 1

			best_pair, freq = self.global_count.most_common(1)[0]
			return best_pair, freq
		else:
			for sentence in ids:
				self.global_count.update(zip(sentence, sentence[1:]))
			best_pair, freq = self.global_count.most_common(1)[0]
			return best_pair, freq

	def merge(self, sentence, pair, idx, sen_idx=-1):
		changed = False
		new_sentence = []
		i = 0
		while i < len(sentence):
			# If we find the pair to merge
			if i < len(sentence) - 1 and sentence[i] == pair[0] and sentence[i + 1] == pair[1]:
				changed = True
				if sen_idx != -1:
					self.vocab_to_sen[idx].add(sen_idx)
				new_sentence.append(idx)
				i += 2  # skip the next token as it's part of the merged pair
			else:
				new_sentence.append(sentence[i])
				i += 1
		return new_sentence, changed

	def encode(self, text: str) -> List[int]:
		tokens = text.encode('utf-8')
		tokens = list(map(int, tokens))
		while True:
			max_pair = None
			max_score = 0
			for pair in zip(tokens, tokens[1:]):
				if pair not in self.pair_in_vocab:
					continue
				if self.scores[pair] > max_score:
					max_score = self.scores[pair]
					max_pair = pair
			if max_pair is None:
				break
			new_token = self.merges[max_pair]
			tokens, _ = self.merge(tokens, max_pair, new_token)
		return tokens

	def decode(self, token_ids: List[int]) -> str:
		tokens = b"".join(self.vocab[idx] for idx in token_ids)
		text = tokens.decode("utf-8", errors="replace")
		return text

	def get_vocab_size(self) -> int:
		"""
		Get the size of the vocabulary

		Returns:
			The number of tokens in the vocabulary
		"""
		return self.vocab_size
