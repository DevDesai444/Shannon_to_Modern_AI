"""
shannon_gen.py

Command-Line Interface for Assignment 3: Approximating Natural Language

Provides CLI commands to:
- analyze: Build frequency tables from text files
- generate: Generate text at different approximation levels
- compare: Compare all 7 levels for an author
- blend: (Bonus) Blend styles from multiple authors

Usage Examples:
    python shannon_gen.py analyze --author austen --file austen_pride_prejudice.txt
    python shannon_gen.py generate --author austen --level word-3 --sentences 5
    python shannon_gen.py generate --author doyle --level word-2 --sentences 3 --anchors elementary,Watson,deduce
    python shannon_gen.py compare --author austen --sentences 2
    python shannon_gen.py blend --authors austen,twain --level word-2 --sentences 3
"""

import argparse
import sys
from pathlib import Path
from generator import TextGenerator


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * 70)


def cmd_analyze(args):
    """
    Command: analyze
    Build frequency tables from a text file
    
    Usage: shannon_gen.py analyze --author austen --file austen_pride_prejudice.txt
    """
    print_header("ANALYZE: Building Frequency Tables")
    
    try:
        from analyze import analyze_author
        
        author = args.author
        filename = args.file
        
        print(f"Author: {author}")
        print(f"Text file: {filename}")
        print()
        
        # Run analysis
        result = analyze_author(author, filename, output_dir='JSON_Files')
        
        if result:
            print_header("ANALYSIS COMPLETE")
            print(f"✅ Successfully analyzed '{author}'")
            print(f"\nGenerated Files:")
            print(f"   • {author}_char_unigrams.json")
            print(f"   • {author}_char_bigrams.json")
            print(f"   • {author}_char_trigrams.json")
            print(f"   • {author}_word_unigrams.json")
            print(f"   • {author}_word_bigrams.json")
            print(f"   • {author}_word_trigrams.json")
            print(f"   • {author}_sentence_stats.json")
            print()
        else:
            print("❌ Analysis failed!")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_generate(args):
    """
    Command: generate
    Generate text using different approximation levels
    
    Usage:
        shannon_gen.py generate --author austen --level char-2 --length 100
        shannon_gen.py generate --author twain --level word-3 --sentences 5
        shannon_gen.py generate --author doyle --level word-2 --sentences 3 --anchors elementary,Watson,deduce
    """
    print_header("GENERATE: Text Generation")
    
    try:
        author = args.author
        level = args.level
        
        print(f"Author: {author}")
        print(f"Level: {level}")
        
        # Determine if character or word level
        is_char_level = level.startswith('char-')
        
        if is_char_level:
            length = args.length if args.length else 100
            print(f"Length: {length} characters")
        else:
            sentences = args.sentences if args.sentences else 3
            print(f"Sentences: {sentences}")
        
        if args.anchors:
            anchor_words = [w.strip() for w in args.anchors.split(',')]
            print(f"Anchor words: {', '.join(anchor_words)}")
        else:
            anchor_words = None
        
        print()
        
        # Initialize generator
        gen = TextGenerator(author, data_dir='.')
        
        # Generate text
        if is_char_level:
            text = gen.generate(level, length=length)
        else:
            text = gen.generate(level, num_sentences=sentences, anchor_words=anchor_words)
        
        # Display result
        print_section("Generated Text")
        print(text)
        print()
    
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_compare(args):
    """
    Command: compare
    Compare all 7 approximation levels for a single author
    
    Usage: shannon_gen.py compare --author austen --sentences 2
    """
    print_header("COMPARE: All Approximation Levels")
    
    try:
        author = args.author
        sentences = args.sentences if args.sentences else 2
        
        print(f"Author: {author}")
        print(f"Sentences per level: {sentences}")
        print()
        
        # Initialize generator
        gen = TextGenerator(author, data_dir='.')
        
        # Define all levels
        levels = [
            'char-0',
            'char-1',
            'char-2',
            'char-3',
            'word-1',
            'word-2',
            'word-3'
        ]
        
        # Generate for each level
        for level in levels:
            print_section(f"Level: {level.upper()}")
            
            try:
                if level.startswith('char-'):
                    text = gen.generate(level, length=100)
                else:
                    text = gen.generate(level, num_sentences=sentences)
                
                # Display text with word wrapping for readability
                display_text = text if len(text) <= 200 else text[:200] + "..."
                print(display_text)
            
            except Exception as e:
                print(f"❌ Error generating {level}: {e}")
        
        print()
        print("="*70)
        print("Comparison complete!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_blend(args):
    """
    Command: blend (BONUS)
    Blend text generation styles from multiple authors
    
    Usage: shannon_gen.py blend --authors austen,twain --level word-2 --sentences 3
    
    Strategy: Generate from each author and blend their styles
    """
    print_header("BLEND: Multiple Author Styles (Bonus Feature)")
    
    try:
        authors_str = args.authors
        level = args.level
        sentences = args.sentences if args.sentences else 3
        
        # Parse authors
        authors = [a.strip() for a in authors_str.split(',')]
        
        print(f"Authors: {', '.join(authors)}")
        print(f"Level: {level}")
        print(f"Sentences per author: {sentences}")
        print()
        
        # Check all authors are valid
        valid_authors = {'austen', 'twain', 'doyle'}
        for author in authors:
            if author not in valid_authors:
                print(f"❌ Invalid author: {author}")
                print(f"   Valid options: austen, twain, doyle")
                sys.exit(1)
        
        if len(authors) < 2:
            print("❌ Blend requires at least 2 authors")
            sys.exit(1)
        
        # Generate from each author
        all_texts = []
        for author in authors:
            print_section(f"Generating from {author.upper()}")
            
            gen = TextGenerator(author, data_dir='.')
            
            if level.startswith('char-'):
                text = gen.generate(level, length=50)
            else:
                text = gen.generate(level, num_sentences=sentences)
            
            all_texts.append(text)
            
            display_text = text if len(text) <= 150 else text[:150] + "..."
            print(display_text)
        
        # Blend strategy: interleave sentences from each author
        print_section("BLENDED OUTPUT")
        
        blended = " ".join(all_texts)
        
        # For word-level, try to create a blend by alternating
        if not level.startswith('char-'):
            sentences_by_author = [t.split('. ') for t in all_texts]
            blended_sentences = []
            
            # Round-robin blend sentences from each author
            max_sentences = max(len(s) for s in sentences_by_author)
            for i in range(max_sentences):
                for j, author_sentences in enumerate(sentences_by_author):
                    if i < len(author_sentences):
                        blended_sentences.append(author_sentences[i])
            
            blended = ". ".join(blended_sentences)
        
        print(blended)
        print()
        
        print("="*70)
        print(f"✅ Blended {len(authors)} authors successfully!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"❌ Error during blending: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for CLI"""
    
    parser = argparse.ArgumentParser(
        description='Shannon Text Generation: n-gram based text approximation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text and build frequency tables
  python shannon_gen.py analyze --author austen --file austen_pride_prejudice.txt

  # Generate text at different levels
  python shannon_gen.py generate --author austen --level char-2 --length 100
  python shannon_gen.py generate --author twain --level word-3 --sentences 5
  python shannon_gen.py generate --author doyle --level word-2 --sentences 3 --anchors elementary,Watson,deduce

  # Compare all approximation levels
  python shannon_gen.py compare --author austen --sentences 2

  # Blend multiple authors (bonus)
  python shannon_gen.py blend --authors austen,twain --level word-2 --sentences 3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ===== ANALYZE COMMAND =====
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze text and build frequency tables'
    )
    analyze_parser.add_argument(
        '--author',
        required=True,
        choices=['austen', 'twain', 'doyle'],
        help='Author to analyze'
    )
    analyze_parser.add_argument(
        '--file',
        required=True,
        help='Path to text file'
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # ===== GENERATE COMMAND =====
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate text at a specified approximation level'
    )
    generate_parser.add_argument(
        '--author',
        required=True,
        choices=['austen', 'twain', 'doyle'],
        help='Author to generate from'
    )
    generate_parser.add_argument(
        '--level',
        required=True,
        choices=['char-0', 'char-1', 'char-2', 'char-3', 'word-1', 'word-2', 'word-3'],
        help='Approximation level'
    )
    generate_parser.add_argument(
        '--length',
        type=int,
        default=100,
        help='Number of characters (for char-level, default: 100)'
    )
    generate_parser.add_argument(
        '--sentences',
        type=int,
        default=3,
        help='Number of sentences (for word-level, default: 3)'
    )
    generate_parser.add_argument(
        '--anchors',
        help='Comma-separated words to include (e.g., "elizabeth,bennet,pride")'
    )
    generate_parser.set_defaults(func=cmd_generate)
    
    # ===== COMPARE COMMAND =====
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare all approximation levels for an author'
    )
    compare_parser.add_argument(
        '--author',
        required=True,
        choices=['austen', 'twain', 'doyle'],
        help='Author to compare'
    )
    compare_parser.add_argument(
        '--sentences',
        type=int,
        default=2,
        help='Number of sentences per level (default: 2)'
    )
    compare_parser.set_defaults(func=cmd_compare)
    
    # ===== BLEND COMMAND (BONUS) =====
    blend_parser = subparsers.add_parser(
        'blend',
        help='(BONUS) Blend styles from multiple authors'
    )
    blend_parser.add_argument(
        '--authors',
        required=True,
        help='Comma-separated author names (e.g., "austen,twain")'
    )
    blend_parser.add_argument(
        '--level',
        required=True,
        choices=['char-0', 'char-1', 'char-2', 'char-3', 'word-1', 'word-2', 'word-3'],
        help='Approximation level'
    )
    blend_parser.add_argument(
        '--sentences',
        type=int,
        default=3,
        help='Number of sentences per author (default: 3)'
    )
    blend_parser.set_defaults(func=cmd_blend)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Display header
    print("\n" + "="*70)
    print("  SHANNON TEXT GENERATION - CLI INTERFACE")
    print("  Assignment 3: Approximating Natural Language")
    print("="*70)
    
    # Execute command if provided
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # No command provided, show help
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
