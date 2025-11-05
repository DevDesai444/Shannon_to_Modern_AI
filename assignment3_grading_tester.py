#!/usr/bin/env python3
"""
CSE 510: Basics of AI - Assignment 3 Grading Tester
Shannon's Information Theory & Text Generation

This script provides automated testing for student implementations.
Students can run this to validate their code before submission.

Usage:
    python assignment3_grading_tester.py

Note: This script should be run in the same environment as the student's
Google Colab notebook (or with the same imports/dependencies).
"""

import math
import json
import sys
import traceback
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

class Assignment3GradingTester:
    def __init__(self):
        self.total_score = 0
        self.max_score = 83
        self.test_results = []
        self.bonus_score = 0
        self.max_bonus = 15
        
    def log_test(self, test_name, passed, score, max_score, details=""):
        """Log a test result"""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'score': score if passed else 0,
            'max_score': max_score,
            'details': details
        })
        if passed:
            self.total_score += score
        print(f"{'✅' if passed else '❌'} {test_name}: {score if passed else 0}/{max_score} points")
        if details:
            print(f"   {details}")
        
    def test_environment_setup(self):
        """Test 1.1: Environment Setup (5 points)"""
        print("\n=== Testing Environment Setup ===")
        
        try:
            # Test required imports
            import nltk
            import random
            import math
            import json
            import requests
            from collections import defaultdict, Counter
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            
            self.log_test("Environment Setup", True, 5, 5, 
                         "All required packages imported successfully")
        except ImportError as e:
            self.log_test("Environment Setup", False, 0, 5, 
                         f"Missing required package: {e}")
    
    def test_shannon_concepts(self, demonstrate_shannon_concepts_func):
        """Test 1.2: Shannon Concepts Demo (5 points)"""
        print("\n=== Testing Shannon Concepts Demo ===")
        
        try:
            # Test if function exists and runs
            demonstrate_shannon_concepts_func()
            self.log_test("Shannon Concepts Demo", True, 5, 5,
                         "Function runs and demonstrates key concepts")
        except Exception as e:
            self.log_test("Shannon Concepts Demo", False, 0, 5,
                         f"Function failed: {e}")
    
    def test_information_analyzer(self, InformationAnalyzer):
        """Test 2.1: Information Analyzer Class (8 points)"""
        print("\n=== Testing Information Analyzer ===")
        
        try:
            analyzer = InformationAnalyzer()
            
            # Test entropy calculation with known values
            test_probs = {'a': 0.5, 'b': 0.3, 'c': 0.2}
            calculated_entropy = analyzer.calculate_entropy(test_probs)
            expected_entropy = -0.5*math.log2(0.5) - 0.3*math.log2(0.3) - 0.2*math.log2(0.2)
            
            if abs(calculated_entropy - expected_entropy) < 0.001:
                score = 5
                details = f"Entropy calculation correct: {calculated_entropy:.3f}"
            else:
                score = 2
                details = f"Entropy calculation error: got {calculated_entropy:.3f}, expected {expected_entropy:.3f}"
            
            # Test perplexity calculation
            calculated_perplexity = analyzer.calculate_perplexity(calculated_entropy)
            expected_perplexity = 2 ** calculated_entropy
            
            if abs(calculated_perplexity - expected_perplexity) < 0.001:
                score += 3
                details += f", Perplexity correct: {calculated_perplexity:.3f}"
            else:
                score += 1
                details += f", Perplexity error: got {calculated_perplexity:.3f}, expected {expected_perplexity:.3f}"
            
            self.log_test("Information Analyzer", score >= 6, score, 8, details)
            
        except Exception as e:
            self.log_test("Information Analyzer", False, 0, 8, f"Class error: {e}")
    
    def test_markov_generator(self, MarkovTextGenerator):
        """Test 1.4-1.6: Markov Text Generators (23 points total)"""
        print("\n=== Testing Markov Text Generators ===")
        
        # Sample frequency data for testing
        sample_1gram = {
            'the': 0.1, 'quick': 0.05, 'brown': 0.03, 'fox': 0.02, 'jumps': 0.02,
            'over': 0.02, 'lazy': 0.02, 'dog': 0.02, 'and': 0.05, 'cat': 0.02
        }
        
        sample_2gram = {
            ('the', 'quick'): 0.01, ('quick', 'brown'): 0.008, ('brown', 'fox'): 0.007,
            ('fox', 'jumps'): 0.006, ('jumps', 'over'): 0.005, ('over', 'the'): 0.004,
            ('the', 'lazy'): 0.003, ('lazy', 'dog'): 0.002, ('dog', 'and'): 0.002,
            ('and', 'cat'): 0.001
        }
        
        sample_3gram = {
            ('the', 'quick', 'brown'): 0.001, ('quick', 'brown', 'fox'): 0.001,
            ('brown', 'fox', 'jumps'): 0.001, ('fox', 'jumps', 'over'): 0.001,
            ('jumps', 'over', 'the'): 0.001, ('over', 'the', 'lazy'): 0.001
        }
        
        orders_to_test = [
            (1, sample_1gram, "Unigram", 8),
            (2, sample_2gram, "Bigram", 8), 
            (3, sample_3gram, "Trigram", 7)
        ]
        
        for order, sample_data, name, max_points in orders_to_test:
            try:
                generator = MarkovTextGenerator(order=order)
                generator.train_from_frequency_data(sample_data)
                
                # Test text generation
                generated_text = generator.generate_text(length=10)
                
                # Basic validation
                words = generated_text.split()
                if len(words) > 0 and len(words) <= 15:  # Allow some flexibility
                    score = max_points
                    details = f"Generated {len(words)} words: '{generated_text[:50]}...'"
                else:
                    score = max_points // 2
                    details = f"Generation length issue: {len(words)} words"
                
                self.log_test(f"{name} Generator", score >= max_points // 2, score, max_points, details)
                
            except Exception as e:
                self.log_test(f"{name} Generator", False, 0, max_points, f"Error: {e}")
    
    def test_creative_generator(self, CreativeTextGenerator):
        """Test 3.1-3.4: Creative Text Generation (27 points total)"""
        print("\n=== Testing Creative Text Generation ===")
        
        try:
            # Sample frequency data for different styles
            frequency_data = {
                'shakespeare_2gram': {('thou', 'art'): 0.01, ('art', 'fair'): 0.008},
                'news_2gram': {('the', 'president'): 0.01, ('president', 'announced'): 0.008},
                'english_1gram': {'the': 0.1, 'quick': 0.05, 'brown': 0.03},
                'english_3gram': {('the', 'quick', 'brown'): 0.001}
            }
            
            creative_gen = CreativeTextGenerator()
            creative_gen.train_style_models(frequency_data)
            
            # Test if models were trained
            if len(creative_gen.style_models) >= 2:
                score = 15
                details = f"Trained {len(creative_gen.style_models)} style models"
            else:
                score = 8
                details = f"Only {len(creative_gen.style_models)} style models trained"
            
            # Test generation for each style
            test_styles = ['shakespeare', 'news'] if 'shakespeare' in creative_gen.style_models else list(creative_gen.style_models.keys())[:2]
            
            generation_works = True
            for style in test_styles[:2]:  # Test max 2 styles
                try:
                    result = creative_gen.generate_creative_text(style, "test prompt", length=10)
                    if 'text' not in result or len(result['text']) == 0:
                        generation_works = False
                        break
                except:
                    generation_works = False
                    break
            
            if generation_works:
                score += 12
                details += ", Style generation working"
            else:
                score += 6
                details += ", Style generation has issues"
            
            self.log_test("Creative Text Generation", score >= 15, score, 27, details)
            
        except Exception as e:
            self.log_test("Creative Text Generation", False, 0, 27, f"Error: {e}")
    
    def test_zipf_analysis(self, analyzer, frequency_data):
        """Test 2.5: Zipf's Law Analysis (5 points)"""
        print("\n=== Testing Zipf's Law Analysis ===")
        
        try:
            # Test with sample frequency data
            sample_freqs = {'the': 0.1, 'of': 0.05, 'and': 0.03, 'to': 0.025, 'a': 0.02}
            
            if hasattr(analyzer, 'analyze_zipf_distribution'):
                zipf_result = analyzer.analyze_zipf_distribution(sample_freqs)
                
                if zipf_result and 'alpha' in zipf_result and 'r_squared' in zipf_result:
                    score = 5
                    details = f"Zipf analysis complete: α={zipf_result['alpha']:.3f}, R²={zipf_result['r_squared']:.3f}"
                else:
                    score = 2
                    details = "Zipf analysis incomplete or missing fields"
            else:
                score = 0
                details = "analyze_zipf_distribution method not found"
            
            self.log_test("Zipf's Law Analysis", score >= 3, score, 5, details)
            
        except Exception as e:
            self.log_test("Zipf's Law Analysis", False, 0, 5, f"Error: {e}")
    
    def test_comprehensive_functionality(self, student_classes):
        """Run comprehensive tests on student implementations"""
        print("🧪 CSE 510 Assignment 3 - Automated Grading Tester")
        print("=" * 60)
        print("Shannon's Information Theory & Text Generation")
        print("=" * 60)
        
        # Test environment setup
        self.test_environment_setup()
        
        # Test Shannon concepts demo
        if 'demonstrate_shannon_concepts' in student_classes:
            self.test_shannon_concepts(student_classes['demonstrate_shannon_concepts'])
        else:
            self.log_test("Shannon Concepts Demo", False, 0, 5, "Function not found")
        
        # Test Information Analyzer
        if 'InformationAnalyzer' in student_classes:
            self.test_information_analyzer(student_classes['InformationAnalyzer'])
        else:
            self.log_test("Information Analyzer", False, 0, 8, "Class not found")
        
        # Test Markov Generator
        if 'MarkovTextGenerator' in student_classes:
            self.test_markov_generator(student_classes['MarkovTextGenerator'])
        else:
            self.log_test("Markov Text Generators", False, 0, 23, "Class not found")
        
        # Test Creative Generator
        if 'CreativeTextGenerator' in student_classes:
            self.test_creative_generator(student_classes['CreativeTextGenerator'])
        else:
            self.log_test("Creative Text Generation", False, 0, 27, "Class not found")
        
        # Test Zipf Analysis (if Information Analyzer exists)
        if 'InformationAnalyzer' in student_classes:
            try:
                analyzer = student_classes['InformationAnalyzer']()
                self.test_zipf_analysis(analyzer, {})
            except:
                self.log_test("Zipf's Law Analysis", False, 0, 5, "Could not instantiate analyzer")
        
        # Code quality assessment (basic)
        self.assess_code_quality(student_classes)
        
        # Print final results
        self.print_final_results()
    
    def assess_code_quality(self, student_classes):
        """Test code quality and documentation (10 points)"""
        print("\n=== Assessing Code Quality ===")
        
        score = 0
        details = []
        
        # Check if classes exist
        required_classes = ['MarkovTextGenerator', 'InformationAnalyzer', 'CreativeTextGenerator']
        existing_classes = sum(1 for cls in required_classes if cls in student_classes)
        
        if existing_classes == len(required_classes):
            score += 4
            details.append("All required classes implemented")
        elif existing_classes >= 2:
            score += 2
            details.append(f"{existing_classes}/{len(required_classes)} required classes found")
        
        # Check for basic error handling (simplified test)
        try:
            if 'MarkovTextGenerator' in student_classes:
                # Test if generator handles empty data gracefully
                gen = student_classes['MarkovTextGenerator'](order=1)
                gen.train_from_frequency_data({})  # Empty data
                score += 3
                details.append("Basic error handling present")
        except:
            score += 1
            details.append("Limited error handling")
        
        # Assume reasonable documentation if classes exist
        if existing_classes >= 2:
            score += 3
            details.append("Code structure appears reasonable")
        
        self.log_test("Code Quality", score >= 6, score, 10, "; ".join(details))
    
    def print_final_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 60)
        print("🎯 FINAL GRADING RESULTS")
        print("=" * 60)
        
        # Category breakdown (actual points awarded)
        categories = {
            'Environment & Setup': 10,      # (5 environment + 5 Shannon demo)
            'Information Analysis': 13,     # (8 analyzer + 5 Zipf analysis)
            'Markov Implementation': 23,    # actual points awarded
            'Creative Generation': 27,      # actual points awarded
            'Code Quality': 10             # actual points awarded
        }
        # Total: 83 points
        
        print(f"\nTotal Score: {self.total_score}/{self.max_score} ({self.total_score/self.max_score*100:.1f}%)")
        
        if self.bonus_score > 0:
            print(f"Bonus Score: {self.bonus_score}/{self.max_bonus}")
            print(f"Final Score: {self.total_score + self.bonus_score}/{self.max_score + self.max_bonus}")
        
        # Grade assignment
        percentage = self.total_score / self.max_score * 100
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        print(f"\nPredicted Grade: {grade}")
        
        # Detailed breakdown
        print(f"\n📊 Test Results Breakdown:")
        print("-" * 50)
        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {result['test']}: {result['score']}/{result['max_score']}")
            if result['details']:
                print(f"   {result['details']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if self.total_score < 60:
            print("   - Review Shannon's information theory fundamentals")
            print("   - Focus on implementing basic Markov chain functionality")
            print("   - Ensure all required classes are properly implemented")
        elif self.total_score < 80:
            print("   - Improve error handling and edge cases")
            print("   - Add more comprehensive testing")
            print("   - Enhance documentation and code comments")
        else:
            print("   - Consider implementing bonus features")
            print("   - Add creative extensions and visualizations")
            print("   - Optimize performance for large datasets")
        
        print(f"\n📋 Submission Checklist:")
        print("   □ Google Colab notebook (.ipynb)")
        print("   □ shannon_analysis_report.json")
        print("   □ generated_samples.txt")
        print("   □ model_comparison.csv")
        print("   □ style_examples.json")
        
        print(f"\n⚠️  Note: This is an automated assessment. Your instructor")
        print(f"   will review your code, analysis, and creativity for final grading.")


def run_student_tests():
    """
    Main function for students to test their implementation
    
    Students should call this function after implementing their classes:
    
    # Example usage in student notebook:
    student_classes = {
        'demonstrate_shannon_concepts': demonstrate_shannon_concepts,
        'InformationAnalyzer': InformationAnalyzer,
        'MarkovTextGenerator': MarkovTextGenerator,
        'CreativeTextGenerator': CreativeTextGenerator
    }
    
    from assignment3_grading_tester import run_student_tests
    run_student_tests(student_classes)
    """
    print("📚 To test your implementation, create a dictionary with your classes:")
    print("   student_classes = {")
    print("       'demonstrate_shannon_concepts': demonstrate_shannon_concepts,")
    print("       'InformationAnalyzer': InformationAnalyzer,")
    print("       'MarkovTextGenerator': MarkovTextGenerator,")
    print("       'CreativeTextGenerator': CreativeTextGenerator")
    print("   }")
    print("   ")
    print("   tester = Assignment3GradingTester()")
    print("   tester.test_comprehensive_functionality(student_classes)")


if __name__ == "__main__":
    print("CSE 510 Assignment 3 - Shannon's Information Theory Grading Tester")
    print("=" * 65)
    run_student_tests()