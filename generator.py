"""
generator.py

TextGenerator class for Assignment 3: Text Generation using n-gram Markov Chains

Implements 7 levels of text generation (char-0 through word-3) based on Shannon's
foundational work on statistical language modeling. Uses frequency tables generated
by analyze.py to create increasingly coherent text.
"""

import json
import random
import statistics
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path


class TextGenerator:
    """
    Generate text using n-gram Markov chains at different orders.
    
    Implements 7 approximation levels:
    - char-0: Random characters (uniform distribution)
    - char-1: Characters with English frequencies
    - char-2: Character bigrams (2nd-order Markov)
    - char-3: Character trigrams (3rd-order Markov)
    - word-1: Word unigrams (random words by frequency)
    - word-2: Word bigrams (2nd-order Markov)
    - word-3: Word trigrams (3rd-order Markov)
    """
    
    def __init__(self, author: str, data_dir: str = 'JSON_Files'):
        """
        Initialize the generator with frequency tables for an author.
        
        Args:
            author: Author name ('austen', 'twain', 'doyle')
            data_dir: Directory containing JSON frequency files
        """
        self.author = author
        self.data_dir = data_dir
        
        # Load all frequency tables
        self.char_unigrams = self._load_json(f'{author}_char_unigrams.json')
        self.char_bigrams = self._load_json(f'{author}_char_bigrams.json')
        self.char_trigrams = self._load_json(f'{author}_char_trigrams.json')
        
        self.word_unigrams = self._load_json(f'{author}_word_unigrams.json')
        self.word_bigrams = self._load_json(f'{author}_word_bigrams.json')
        self.word_trigrams = self._load_json(f'{author}_word_trigrams.json')
        
        self.sentence_stats = self._load_json(f'{author}_sentence_stats.json')
        
        # Convert frequency tables to probabilities
        self._build_ngram_models()
    
    def _load_json(self, filename: str) -> dict:
        """Load JSON file from data directory."""
        filepath = Path(self.data_dir) / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: {filename} not found")
            return {}
    
    def _build_ngram_models(self):
        """Convert frequency tables to probability models and Markov chains."""
        
        # Character models
        self.char_unigram_probs = self._frequencies_to_probabilities(self.char_unigrams)
        self.char_bigram_chains = self._build_markov_chains(self.char_bigrams)
        self.char_trigram_chains = self._build_markov_chains(self.char_trigrams)
        
        # Word models
        self.word_unigram_probs = self._frequencies_to_probabilities(self.word_unigrams)
        self.word_bigram_chains = self._build_markov_chains(self.word_bigrams)
        self.word_trigram_chains = self._build_markov_chains(self.word_trigrams)
    
    def _frequencies_to_probabilities(self, frequencies: Dict[str, int]) -> Dict[str, float]:
        """Convert frequency counts to probability distribution."""
        if not frequencies:
            return {}
        
        total = sum(frequencies.values())
        return {key: count / total for key, count in frequencies.items()}
    
    def _build_markov_chains(self, ngrams: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Build Markov chains from n-gram frequencies.
        
        For bigrams: {key1||key2} -> chains[key1][key2] = probability
        For trigrams: {key1||key2||key3} -> chains[(key1, key2)][key3] = probability
        """
        chains = defaultdict(lambda: defaultdict(float))
        
        for ngram_str, count in ngrams.items():
            parts = ngram_str.split('||')
            
            if len(parts) == 2:
                # Bigram: prefix is first part, next is second part
                prefix = parts[0]
                next_item = parts[1]
                chains[prefix][next_item] += count
            
            elif len(parts) == 3:
                # Trigram: prefix is tuple of first two, next is third
                prefix = (parts[0], parts[1])
                next_item = parts[2]
                chains[prefix][next_item] += count
        
        # Normalize to probabilities
        for prefix in chains:
            total = sum(chains[prefix].values())
            for next_item in chains[prefix]:
                chains[prefix][next_item] /= total
        
        return dict(chains)
    
    def _weighted_choice(self, choices: Dict[str, float]) -> Optional[str]:
        """Select an item using weighted probability distribution."""
        if not choices:
            return None
        
        items = list(choices.keys())
        weights = list(choices.values())
        
        try:
            return random.choices(items, weights=weights, k=1)[0]
        except:
            return random.choice(items) if items else None
    
    def _sample_sentence_length(self) -> int:
        """Sample a sentence length from the author's distribution."""
        if not self.sentence_stats or 'length_distribution' not in self.sentence_stats:
            return random.randint(5, 20)  # Fallback default
        
        length_dist = self.sentence_stats['length_distribution']
        
        # Convert string keys back to integers
        lengths = []
        for length_str, count in length_dist.items():
            lengths.extend([int(length_str)] * count)
        
        if lengths:
            return random.choice(lengths)
        return random.randint(5, 20)
    
    # ========== CHARACTER-LEVEL GENERATION ==========
    
    def generate_char_0(self, length: int = 100) -> str:
        """
        Zero-order character generation: random characters with equal probability.
        All 26 letters + space treated with equal likelihood.
        """
        chars = 'abcdefghijklmnopqrstuvwxyz '
        return ''.join(random.choice(chars) for _ in range(length))
    
    def generate_char_1(self, length: int = 100) -> str:
        """
        First-order character generation: use English character frequencies.
        Each character chosen independently based on frequency distribution.
        """
        result = []
        for _ in range(length):
            char = self._weighted_choice(self.char_unigram_probs)
            if char:
                result.append(char)
        
        return ''.join(result)
    
    def generate_char_2(self, length: int = 100) -> str:
        """
        Second-order character generation: bigram-based Markov chain.
        Each character depends on the previous character (bigram statistics).
        """
        if not self.char_bigram_chains:
            return self.generate_char_1(length)
        
        result = []
        
        # Start with a random character from unigrams
        current = self._weighted_choice(self.char_unigram_probs)
        if current:
            result.append(current)
        
        # Generate remaining characters using bigram chains
        while len(result) < length:
            next_char = self.char_bigram_chains.get(current, {})
            next_char = self._weighted_choice(next_char)
            
            if next_char is None:
                # Fallback: pick random character
                next_char = self._weighted_choice(self.char_unigram_probs)
            
            if next_char:
                result.append(next_char)
                current = next_char
            else:
                break
        
        return ''.join(result[:length])
    
    def generate_char_3(self, length: int = 100) -> str:
        """
        Third-order character generation: trigram-based Markov chain.
        Each character depends on the previous two characters (trigram statistics).
        """
        if not self.char_trigram_chains:
            return self.generate_char_2(length)
        
        result = []
        
        # Start with first two characters
        char1 = self._weighted_choice(self.char_unigram_probs)
        char2 = self._weighted_choice(self.char_unigram_probs)
        
        if char1:
            result.append(char1)
        if char2:
            result.append(char2)
        
        # Generate remaining characters using trigram chains
        while len(result) < length:
            context = (result[-2], result[-1])
            
            next_char_probs = self.char_trigram_chains.get(context, {})
            next_char = self._weighted_choice(next_char_probs)
            
            if next_char is None:
                # Fallback to bigram
                next_char = self.char_bigram_chains.get(result[-1], {})
                next_char = self._weighted_choice(next_char)
            
            if next_char is None:
                # Fallback to unigram
                next_char = self._weighted_choice(self.char_unigram_probs)
            
            if next_char:
                result.append(next_char)
            else:
                break
        
        return ''.join(result[:length])
    
    # ========== WORD-LEVEL GENERATION ==========
    
    def generate_word_1(self, num_sentences: int = 3, anchor_words: Optional[List[str]] = None) -> str:
        """
        First-order word generation: unigram-based.
        Words chosen independently based on frequency distribution.
        """
        sentences = []
        attempts = 0
        max_attempts = 100
        
        while len(sentences) < num_sentences and attempts < max_attempts:
            sentence_length = self._sample_sentence_length()
            words = []
            
            for _ in range(sentence_length):
                word = self._weighted_choice(self.word_unigram_probs)
                if word:
                    words.append(word)
            
            if words:
                # Format sentence: capitalize first word, add period
                sentence = ' '.join(words)
                sentence = sentence[0].upper() + sentence[1:] + '.'
                sentences.append(sentence)
            
            attempts += 1
        
        # Check for anchor words
        result = ' '.join(sentences)
        
        if anchor_words:
            result = self._integrate_anchor_words(result, anchor_words, num_sentences)
        
        return result
    
    def generate_word_2(self, num_sentences: int = 3, anchor_words: Optional[List[str]] = None) -> str:
        """
        Second-order word generation: bigram-based Markov chain.
        Word pairs drive the generation (2nd-order Markov).
        """
        if not self.word_bigram_chains:
            return self.generate_word_1(num_sentences, anchor_words)
        
        sentences = []
        
        for _ in range(num_sentences):
            sentence_length = self._sample_sentence_length()
            words = []
            
            # Start with a random word
            current_word = self._weighted_choice(self.word_unigram_probs)
            if current_word:
                words.append(current_word)
            
            # Generate remaining words using bigram chains
            while len(words) < sentence_length:
                next_word_probs = self.word_bigram_chains.get(current_word, {})
                next_word = self._weighted_choice(next_word_probs)
                
                if next_word is None:
                    # Fallback: pick random word
                    next_word = self._weighted_choice(self.word_unigram_probs)
                
                if next_word:
                    words.append(next_word)
                    current_word = next_word
                else:
                    break
            
            if words:
                # Format sentence
                sentence = ' '.join(words)
                sentence = sentence[0].upper() + sentence[1:] + '.'
                sentences.append(sentence)
        
        result = ' '.join(sentences)
        
        if anchor_words:
            result = self._integrate_anchor_words(result, anchor_words, num_sentences)
        
        return result
    
    def generate_word_3(self, num_sentences: int = 3, anchor_words: Optional[List[str]] = None) -> str:
        """
        Third-order word generation: trigram-based Markov chain.
        Word triplets drive the generation (3rd-order Markov).
        """
        if not self.word_trigram_chains:
            return self.generate_word_2(num_sentences, anchor_words)
        
        sentences = []
        
        for _ in range(num_sentences):
            sentence_length = self._sample_sentence_length()
            words = []
            
            # Start with first two words
            word1 = self._weighted_choice(self.word_unigram_probs)
            word2 = self._weighted_choice(self.word_unigram_probs)
            
            if word1:
                words.append(word1)
            if word2:
                words.append(word2)
            
            # Generate remaining words using trigram chains
            while len(words) < sentence_length:
                context = (words[-2], words[-1])
                
                next_word_probs = self.word_trigram_chains.get(context, {})
                next_word = self._weighted_choice(next_word_probs)
                
                if next_word is None:
                    # Fallback to bigram
                    next_word_probs = self.word_bigram_chains.get(words[-1], {})
                    next_word = self._weighted_choice(next_word_probs)
                
                if next_word is None:
                    # Fallback to unigram
                    next_word = self._weighted_choice(self.word_unigram_probs)
                
                if next_word:
                    words.append(next_word)
                else:
                    break
            
            if words:
                # Format sentence
                sentence = ' '.join(words)
                sentence = sentence[0].upper() + sentence[1:] + '.'
                sentences.append(sentence)
        
        result = ' '.join(sentences)
        
        if anchor_words:
            result = self._integrate_anchor_words(result, anchor_words, num_sentences)
        
        return result
    
    # ========== ANCHOR WORD INTEGRATION ==========
    
    def _integrate_anchor_words(self, text: str, anchor_words: List[str], num_sentences: int) -> str:
        """
        Integrate anchor words naturally into generated text.
        
        Strategy: Generate multiple candidates and select ones containing anchor words.
        If an anchor word is missing, replace a sentence with text containing it.
        """
        if not anchor_words:
            return text
        
        # Check which anchor words are in the text
        text_lower = text.lower()
        missing_anchors = [w for w in anchor_words if w.lower() not in text_lower]
        
        if not missing_anchors:
            return text  # All anchor words already present
        
        # Regenerate sentences containing missing anchor words
        sentences = text.split('. ')
        
        for anchor in missing_anchors:
            # Try to generate a sentence containing this anchor word
            for attempt in range(10):
                if anchor in self.word_unigrams:
                    # Generate around this word
                    candidate = self._generate_around_anchor(anchor, num_words=self._sample_sentence_length())
                    if candidate and anchor.lower() in candidate.lower():
                        # Replace a random sentence
                        idx = random.randint(0, len(sentences) - 1)
                        sentences[idx] = candidate
                        break
        
        return '. '.join(sentences)
    
    def _generate_around_anchor(self, anchor_word: str, num_words: int = 10) -> Optional[str]:
        """
        Generate a sentence that naturally includes a specific anchor word.
        """
        words = []
        current = anchor_word
        words.append(current)
        
        # Generate words before the anchor
        before_count = random.randint(2, 5)
        prev_words = [anchor_word]
        
        for _ in range(before_count):
            if len(prev_words) >= 2:
                context = (prev_words[-2], prev_words[-1])
                next_word = self._weighted_choice(self.word_bigram_chains.get(context, {}))
            else:
                next_word = self._weighted_choice(self.word_unigram_probs)
            
            if next_word:
                prev_words.append(next_word)
        
        # Generate words after the anchor
        after_count = num_words - before_count
        next_words = [anchor_word]
        
        for _ in range(after_count):
            if len(next_words) >= 2:
                context = (next_words[-2], next_words[-1])
                word = self._weighted_choice(self.word_bigram_chains.get(context, {}))
            else:
                word = self._weighted_choice(self.word_unigram_probs)
            
            if word:
                next_words.append(word)
        
        # Combine and format
        all_words = prev_words[::-1] + next_words[1:]
        if all_words:
            sentence = ' '.join(all_words)
            sentence = sentence[0].upper() + sentence[1:] + '.'
            return sentence
        
        return None
    
    # ========== MAIN GENERATION INTERFACE ==========
    
    def generate(self, level: str, length: int = 100, num_sentences: int = 3, 
                 anchor_words: Optional[List[str]] = None) -> str:
        """
        Generate text at a specified approximation level.
        
        Args:
            level: Generation level ('char-0' through 'word-3')
            length: Number of characters (for character-level)
            num_sentences: Number of sentences (for word-level)
            anchor_words: List of words to include naturally
        
        Returns:
            Generated text string
        """
        level_map = {
            'char-0': lambda: self.generate_char_0(length),
            'char-1': lambda: self.generate_char_1(length),
            'char-2': lambda: self.generate_char_2(length),
            'char-3': lambda: self.generate_char_3(length),
            'word-1': lambda: self.generate_word_1(num_sentences, anchor_words),
            'word-2': lambda: self.generate_word_2(num_sentences, anchor_words),
            'word-3': lambda: self.generate_word_3(num_sentences, anchor_words),
        }
        
        if level not in level_map:
            raise ValueError(f"Invalid level: {level}. Must be 'char-0' through 'word-3'")

        return level_map[level]()