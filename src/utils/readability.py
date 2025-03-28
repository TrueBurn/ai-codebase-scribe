from typing import Dict, List, TypedDict, Optional
import re
from textstat import textstat

class ReadabilityMetrics(TypedDict):
    """TypedDict for readability metrics."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    avg_sentence_length: float
    complex_word_percentage: float
    paragraphs: int
    overall: float  # Overall readability score

class ReadabilityScorer:
    """
    Analyzes and scores documentation readability.
    
    This class uses various readability metrics to evaluate text complexity:
    - Flesch Reading Ease: Higher scores (100+) indicate easier text, lower scores indicate complex text
    - Flesch-Kincaid Grade: Indicates the US grade level needed to understand the text
    - Gunning Fog Index: Estimates years of formal education needed to understand the text
    - Average Sentence Length: Longer sentences are typically harder to read
    - Complex Word Percentage: Higher percentage of complex words (3+ syllables) increases difficulty
    """
    
    GRADE_LEVELS = {
        'easy': 8.0,      # 8th grade reading level
        'medium': 12.0,   # 12th grade reading level
        'hard': 16.0      # College level
    }
    
    # Default thresholds for readability recommendations
    DEFAULT_THRESHOLDS = {
        'flesch_reading_ease': 60.0,  # Below this is considered difficult
        'avg_sentence_length': 25.0,  # Above this is considered too long
        'complex_word_percentage': 20.0,  # Above this is considered too complex
        'min_paragraphs': 2  # Minimum recommended paragraphs
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the ReadabilityScorer.
        
        Args:
            thresholds: Optional dictionary of custom thresholds to override defaults
        """
        self.scores: Dict[str, ReadabilityMetrics] = {}
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def analyze_text(self, text: str, section_name: str) -> ReadabilityMetrics:
        """
        Analyze text and return readability metrics.
        
        Args:
            text: The text to analyze
            section_name: Identifier for this text section
            
        Returns:
            Dictionary of readability metrics including an overall score
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text:
            raise ValueError("Cannot analyze empty text")
        
        # Calculate basic metrics
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        avg_sentence_length = self._average_sentence_length(text)
        complex_word_percentage = self._complex_word_percentage(text)
        paragraphs = len(text.split('\n\n'))
        
        # Calculate overall score (higher means more complex/difficult)
        # Normalize and weight different metrics to create a composite score
        overall = (
            (100 - min(flesch_reading_ease, 100)) * 0.4 +  # Invert so higher = more complex
            min(flesch_kincaid_grade * 3, 50) * 0.3 +      # Scale grade level
            min(complex_word_percentage, 50) * 0.3          # Complex word percentage
        )
        
        metrics: ReadabilityMetrics = {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'gunning_fog': gunning_fog,
            'avg_sentence_length': avg_sentence_length,
            'complex_word_percentage': complex_word_percentage,
            'paragraphs': paragraphs,
            'overall': overall
        }
        
        # Store scores for later reference
        self.scores[section_name] = metrics
        return metrics
    
    def get_recommendations(self, section_name: str) -> List[str]:
        """
        Get improvement recommendations based on scores.
        
        Args:
            section_name: Identifier for the text section
            
        Returns:
            List of recommendation strings
            
        Raises:
            KeyError: If section_name hasn't been analyzed yet
        """
        if section_name not in self.scores:
            raise KeyError(f"No analysis found for section '{section_name}'. Run analyze_text first.")
        
        metrics = self.scores[section_name]
        recommendations = []
        
        if metrics['flesch_reading_ease'] < self.thresholds['flesch_reading_ease']:
            recommendations.append("Consider simplifying language for better readability")
        
        if metrics['avg_sentence_length'] > self.thresholds['avg_sentence_length']:
            recommendations.append("Try breaking up longer sentences")
        
        if metrics['complex_word_percentage'] > self.thresholds['complex_word_percentage']:
            recommendations.append("Use simpler words where possible")
        
        if metrics['paragraphs'] < self.thresholds['min_paragraphs']:
            recommendations.append("Consider breaking content into more paragraphs")
        
        return recommendations
    
    def _average_sentence_length(self, text: str) -> float:
        """
        Calculate average sentence length.
        
        Args:
            text: The text to analyze
            
        Returns:
            Average number of words per sentence
        """
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0
        
        words = len(text.split())
        return words / len(sentences)
    
    def _complex_word_percentage(self, text: str) -> float:
        """
        Calculate percentage of complex words (3+ syllables).
        
        Args:
            text: The text to analyze
            
        Returns:
            Percentage of words that are complex (3+ syllables)
        """
        words = text.split()
        if not words:
            return 0
        
        complex_words = sum(1 for word in words if textstat.syllable_count(word) >= 3)
        return (complex_words / len(words) * 100)
    
    def is_section_readable(self, section_name: str, target_level: str = 'medium') -> bool:
        """
        Check if section meets target readability level.
        
        Args:
            section_name: Identifier for the text section
            target_level: Target readability level ('easy', 'medium', or 'hard')
            
        Returns:
            True if the section meets the target readability level, False otherwise
            
        Raises:
            KeyError: If section_name hasn't been analyzed yet or target_level is invalid
        """
        if section_name not in self.scores:
            raise KeyError(f"No analysis found for section '{section_name}'. Run analyze_text first.")
        
        if target_level not in self.GRADE_LEVELS:
            raise KeyError(f"Invalid target level '{target_level}'. Must be one of: {', '.join(self.GRADE_LEVELS.keys())}")
        
        metrics = self.scores[section_name]
        target_grade = self.GRADE_LEVELS[target_level]
        
        return (
            metrics['flesch_kincaid_grade'] <= target_grade and
            metrics['gunning_fog'] <= target_grade + 2
        )
    
    def get_overall_score(self, section_name: str) -> float:
        """
        Get the overall readability score for a section.
        
        Args:
            section_name: Identifier for the text section
            
        Returns:
            Overall readability score (higher means more complex/difficult)
            
        Raises:
            KeyError: If section_name hasn't been analyzed yet
        """
        if section_name not in self.scores:
            raise KeyError(f"No analysis found for section '{section_name}'. Run analyze_text first.")
        
        return self.scores[section_name]['overall']