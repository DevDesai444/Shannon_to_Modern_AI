"""
analyze.py

Complete text analysis script for Assignment 3: Approximating Natural Language

This script analyzes three literary texts (Austen, Twain, Doyle) and computes:
- Character-level n-grams (unigrams, bigrams, trigrams)
- Word-level n-grams (unigrams, bigrams, trigrams)
- Sentence structure statistics (length distribution, mean, std dev)

All results are saved as JSON files for later use in text generation.
"""

import json
import statistics
from pathlib import Path
from starter_preprocess import TextPreprocessor, FrequencyAnalyzer


def analyze_author(author_name: str, filename: str, output_dir: str = '.') -> dict:
    """
    Complete analysis pipeline for one author.
    
    Args:
        author_name: Short name (e.g., 'austen', 'twain', 'doyle')
        filename: Path to text file
        output_dir: Directory to save JSON output files
    
    Returns:
        Dictionary with analysis summary
    """
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {author_name.upper()}")
    print(f"{'='*70}")
    
    # Initialize preprocessor and analyzer
    preprocessor = TextPreprocessor()
    analyzer = FrequencyAnalyzer()
    
    # ========== STEP 1: LOAD AND CLEAN TEXT ==========
    print(f"[1/4] Loading and cleaning text from {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: File '{filename}' not found!")
        return None
    
    # Remove Project Gutenberg headers/footers
    clean_text = preprocessor.clean_gutenberg_text(raw_text)
    
    # Normalize text while preserving sentence boundaries
    normalized_text = preprocessor.normalize_text(clean_text, preserve_sentences=True)
    
    text_length = len(normalized_text)
    print(f"   ✓ Text length: {text_length:,} characters")
    
    # ========== STEP 2: CHARACTER-LEVEL ANALYSIS ==========
    print(f"[2/4] Computing character-level n-grams...")
    
    # Tokenize into characters
    char_tokens = preprocessor.tokenize_chars(normalized_text, include_space=True)
    print(f"   ✓ Total characters: {len(char_tokens):,}")
    
    # Calculate n-grams
    char_unigrams = analyzer.calculate_ngrams(char_tokens, n=1)
    char_bigrams = analyzer.calculate_ngrams(char_tokens, n=2)
    char_trigrams = analyzer.calculate_ngrams(char_tokens, n=3)
    
    print(f"   ✓ Unique char unigrams: {len(char_unigrams):,}")
    print(f"   ✓ Unique char bigrams: {len(char_bigrams):,}")
    print(f"   ✓ Unique char trigrams: {len(char_trigrams):,}")
    
    # Save character frequencies
    analyzer.save_frequencies(char_unigrams, f'{output_dir}/{author_name}_char_unigrams.json')
    analyzer.save_frequencies(char_bigrams, f'{output_dir}/{author_name}_char_bigrams.json')
    analyzer.save_frequencies(char_trigrams, f'{output_dir}/{author_name}_char_trigrams.json')
    
    print(f"   ✓ Saved character frequency files")
    
    # ========== STEP 3: WORD-LEVEL ANALYSIS ==========
    print(f"[3/4] Computing word-level n-grams...")
    
    # Tokenize into words and sentences
    sentences = preprocessor.tokenize_sentences(normalized_text)
    word_tokens = preprocessor.tokenize_words(normalized_text)
    
    print(f"   ✓ Total sentences: {len(sentences):,}")
    print(f"   ✓ Total words: {len(word_tokens):,}")
    
    # Calculate n-grams
    word_unigrams = analyzer.calculate_ngrams(word_tokens, n=1)
    word_bigrams = analyzer.calculate_ngrams(word_tokens, n=2)
    word_trigrams = analyzer.calculate_ngrams(word_tokens, n=3)
    
    print(f"   ✓ Unique word unigrams: {len(word_unigrams):,}")
    print(f"   ✓ Unique word bigrams: {len(word_bigrams):,}")
    print(f"   ✓ Unique word trigrams: {len(word_trigrams):,}")
    
    # Save word frequencies
    analyzer.save_frequencies(word_unigrams, f'{output_dir}/{author_name}_word_unigrams.json')
    analyzer.save_frequencies(word_bigrams, f'{output_dir}/{author_name}_word_bigrams.json')
    analyzer.save_frequencies(word_trigrams, f'{output_dir}/{author_name}_word_trigrams.json')
    
    print(f"   ✓ Saved word frequency files")
    
    # ========== STEP 4: SENTENCE STRUCTURE ANALYSIS ==========
    print(f"[4/4] Computing sentence structure statistics...")
    
    # Get sentence lengths (in words)
    sentence_lengths = preprocessor.get_sentence_lengths(sentences)
    
    # Handle edge cases
    if len(sentence_lengths) == 0:
        print("❌ ERROR: No sentences found!")
        return None
    
    # Compute statistics
    avg_length = statistics.mean(sentence_lengths)
    std_length = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    
    # Create length distribution dictionary
    length_distribution = {}
    for length in sentence_lengths:
        length_str = str(length)
        length_distribution[length_str] = length_distribution.get(length_str, 0) + 1
    
    # Compile sentence statistics
    sentence_stats = {
        'total_sentences': len(sentences),
        'average_length': round(avg_length, 2),
        'std_dev_length': round(std_length, 2),
        'min_length': min_length,
        'max_length': max_length,
        'length_distribution': length_distribution
    }
    
    print(f"   ✓ Total sentences: {sentence_stats['total_sentences']:,}")
    print(f"   ✓ Average sentence length: {sentence_stats['average_length']} words")
    print(f"   ✓ Std dev sentence length: {sentence_stats['std_dev_length']}")
    print(f"   ✓ Sentence length range: {min_length} - {max_length} words")
    
    # Save sentence statistics
    with open(f'{output_dir}/{author_name}_sentence_stats.json', 'w', encoding='utf-8') as f:
        json.dump(sentence_stats, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Saved sentence statistics")
    
    # ========== SUMMARY ==========
    summary = {
        'author': author_name,
        'char_unigrams': len(char_unigrams),
        'char_bigrams': len(char_bigrams),
        'char_trigrams': len(char_trigrams),
        'word_unigrams': len(word_unigrams),
        'word_bigrams': len(word_bigrams),
        'word_trigrams': len(word_trigrams),
        'total_sentences': len(sentences),
        'avg_sentence_length': avg_length,
        'total_words': len(word_tokens),
        'total_chars': len(char_tokens)
    }
    
    return summary


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("CSE 510 - ASSIGNMENT 3: APPROXIMATING NATURAL LANGUAGE")
    print("Part 2: Statistical Analysis")
    print("="*70)
    
    # Define authors and their text files
    authors = {
        'austen': 'austen_pride_prejudice.txt',
        'twain': 'twain_tom_sawyer.txt',
        'doyle': 'doyle_sherlock_holmes.txt'
    }
    
    # Create output directory if it doesn't exist
    output_dir = '.'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze each author
    results = {}
    for author_name, filename in authors.items():
        result = analyze_author(author_name, filename, output_dir)
        if result:
            results[author_name] = result
    
    # ========== FINAL SUMMARY ==========
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    if results:
        print(f"\n✅ Successfully analyzed {len(results)} texts\n")
        
        # Print comparison table
        print(f"{'Author':<12} {'Sentences':<12} {'Avg Len':<12} {'Unique Words':<15}")
        print("-" * 52)
        for author, data in results.items():
            print(f"{author:<12} {data['total_sentences']:<12,} {data['avg_sentence_length']:<12.2f} {data['word_unigrams']:<15,}")
        
        # List generated files
        print(f"\n📁 Generated JSON files:")
        print(f"   • {len(results)} × 7 frequency files (char/word unigrams, bigrams, trigrams)")
        print(f"   • {len(results)} × 1 sentence statistics file")
        print(f"   • Total: {len(results) * 8} JSON files created")
        
        print(f"\n✨ All analyses complete! Files are ready for Part 3 (Text Generation)")
    else:
        print("❌ No texts were successfully analyzed.")


if __name__ == '__main__':
    main()