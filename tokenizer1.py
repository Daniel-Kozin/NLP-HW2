from typing import List

from base_tokenizer import BaseTokenizer


class Tokenizer1(BaseTokenizer):

	def train(self, texts: List[str]) -> None:
	# Train your tokenizer on the given texts
		pass

	def encode(self, text: str) -> List[int]:
		# Convert text to token IDs
		pass

	def decode(self, token_ids: List[int]) -> str:
		# Convert token IDs back to text
		pass