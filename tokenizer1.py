from typing import List
from collections import Counter
from base_tokenizer import BaseTokenizer


class Tokenizer1(BaseTokenizer):

	def __init__(self, vocab_size):
		super(Tokenizer1, self).__init__()
		self.merges = {}
		self.vocab_size = vocab_size
		self.vocab = {idx: bytes([idx]) for idx in range(256)}
		self.scores = {}
		self.global_count = Counter()

	def train(self, texts: List[str]) -> None:
		num_merges = self.vocab_size - 256
		tokens = self.encode_batch(texts)  # List[List[int]]
		ids = list(tokens)  # copy so we don't destroy the original list
		changed = None
		pair = None

		for i in range(num_merges):
			if i % 50 == 0:
				print(f"Iteration {i}")
				self.global_count = Counter({k: v for k, v in self.global_count.items() if v > 0})

			idx = 256 + i
			pair, count_pair = self.get_highest_count(ids, changed, idx - 1, pair)  # Find the best pair
			self.scores[pair] = count_pair  # Remember that pair score
			# Add this pair to the vocab in his original form
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
			self.merges[pair] = idx

			print(f"merging {pair} into a new token {idx}")
			# Merge the new pair into the text
			ids, changed = self.merge(ids, pair, idx)

	def get_highest_count(self, ids, changed, new_token, pair):
		if changed is not None:
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

	def merge(self, ids, pair, idx):
		new_ids = []
		sen_changed = []
		for sentence in ids:
			changed = False
			new_sentence = []
			i = 0
			while i < len(sentence):
				# If we find the pair to merge
				if i < len(sentence) - 1 and sentence[i] == pair[0] and sentence[i + 1] == pair[1]:
					changed = True
					new_sentence.append(idx)
					i += 2  # skip the next token as it's part of the merged pair
				else:
					new_sentence.append(sentence[i])
					i += 1
			new_ids.append(new_sentence)
			sen_changed.append(changed)
		return new_ids, sen_changed

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
