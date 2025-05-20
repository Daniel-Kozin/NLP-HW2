from typing import List

from base_tokenizer import BaseTokenizer


class Tokenizer1(BaseTokenizer):

	def __init__(self, vocab_size):
		super(Tokenizer1, self).__init__()
		self.vocab = None
		self.merges = None
		self.vocab_size = vocab_size

	def train(self, texts: List[str]) -> None:
		num_merges = self.vocab_size - 256

		tokens = self.encode_batch(texts)
		ids = list(tokens)  # copy so we don't destroy the original list

		self.merges = {}  # (int, int) -> int
		for i in range(num_merges):
			stats = self.get_stats(ids)
			pair = max(stats, key=stats.get)
			idx = 256 + i
			print(f"merging {pair} into a new token {idx}")
			ids = self.merge(ids, pair, idx)
			self.merges[pair] = idx

		self.vocab = {idx: bytes([idx]) for idx in range(256)}
		for (p0, p1), idx in self.merges.items():
			self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

	def get_stats(self, ids):
		counts = {}
		for pair in zip(ids, ids[1:]):
			counts[pair] = counts.get(pair, 0) + 1 #a
		return counts

	def merge(self, ids, pair, idx):
		newids = []
		i = 0
		while i < len(ids):
			if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
				newids.append(idx)
				i += 2
			else:
				newids.append(ids[i])
				i += 1
		return newids

	def encode(self, text: str) -> List[int]:
		tokens = text.encode("utf-8")  # raw bytes
		tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience
		return tokens

	def decode(self, token_ids: List[int]) -> str:
		tokens = b"".join(self.vocab[idx] for idx in token_ids)
		text = tokens.decode("utf-8", errors="replace")
		return text
