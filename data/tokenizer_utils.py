from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []
        for num in numbers:
            text = str(num)
            tokens = self.greedy_tokenizer(text, vocab)
            res.append(tokens)
        return res

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self.greedy_tokenizer(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.greedy_tokenizer(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)

    def greedy_tokenizer(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            longest = None
            for length in range(len(text), i, -1):
                if text[i:length] in vocab:
                    longest = text[i:length]
                    break
            if longest is None:
                tokens.append(text[i])
                i+=1
            else:
                tokens.append(longest)
                i+=len(longest)
        return tokens









