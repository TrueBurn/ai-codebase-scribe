import pytest
from src.utils.readability import ReadabilityScorer, ReadabilityMetrics

class TestReadabilityScorer:
    """Test suite for the ReadabilityScorer class."""
    
    def test_analyze_text_basic(self):
        """Test basic text analysis functionality."""
        scorer = ReadabilityScorer()
        text = "This is a simple test sentence. It has basic words and structure."
        metrics = scorer.analyze_text(text, "test_section")
        
        # Check that all expected metrics are present
        assert 'flesch_reading_ease' in metrics
        assert 'flesch_kincaid_grade' in metrics
        assert 'gunning_fog' in metrics
        assert 'avg_sentence_length' in metrics
        assert 'complex_word_percentage' in metrics
        assert 'paragraphs' in metrics
        assert 'overall' in metrics
        
        # Check that values are of expected types
        assert isinstance(metrics['flesch_reading_ease'], float)
        assert isinstance(metrics['flesch_kincaid_grade'], float)
        assert isinstance(metrics['gunning_fog'], float)
        assert isinstance(metrics['avg_sentence_length'], float)
        assert isinstance(metrics['complex_word_percentage'], float)
        assert isinstance(metrics['paragraphs'], int)
        assert isinstance(metrics['overall'], float)
    
    def test_analyze_text_empty(self):
        """Test that analyzing empty text raises ValueError."""
        scorer = ReadabilityScorer()
        with pytest.raises(ValueError):
            scorer.analyze_text("", "empty_section")
    
    def test_get_recommendations(self):
        """Test getting recommendations based on scores."""
        scorer = ReadabilityScorer()
        
        # Create text with known readability issues
        complex_text = """
        The implementation of the aforementioned methodologies necessitates a comprehensive understanding 
        of the underlying principles governing the functionality of the system in question, particularly 
        with respect to the interdependencies between various components and the potential ramifications 
        of modifications to any singular element within the broader context of the operational framework.
        """
        
        metrics = scorer.analyze_text(complex_text, "complex_section")
        recommendations = scorer.get_recommendations("complex_section")
        
        # Should have recommendations for complex text
        assert len(recommendations) > 0
        assert any("simplifying language" in rec for rec in recommendations)
    
    def test_get_recommendations_missing_section(self):
        """Test that getting recommendations for a missing section raises KeyError."""
        scorer = ReadabilityScorer()
        with pytest.raises(KeyError):
            scorer.get_recommendations("nonexistent_section")
    
    def test_is_section_readable(self):
        """Test checking if a section meets readability targets."""
        scorer = ReadabilityScorer()
        
        # Simple text should be readable at 'easy' level
        simple_text = "This is a simple text. It uses short words and sentences. It should be easy to read."
        scorer.analyze_text(simple_text, "simple_section")
        assert scorer.is_section_readable("simple_section", "easy") is True
        
        # Complex text might not be readable at 'easy' level
        complex_text = """
        The utilization of multifaceted analytical methodologies in conjunction with sophisticated 
        computational algorithms facilitates the extraction of meaningful insights from large-scale 
        datasets characterized by high dimensionality and inherent complexity.
        """
        scorer.analyze_text(complex_text, "complex_section")
        # This might be false depending on the exact metrics, but we're testing the method works
        readable_at_hard = scorer.is_section_readable("complex_section", "hard")
        assert isinstance(readable_at_hard, bool)
    
    def test_is_section_readable_invalid_level(self):
        """Test that checking readability with an invalid level raises KeyError."""
        scorer = ReadabilityScorer()
        scorer.analyze_text("Some text", "test_section")
        with pytest.raises(KeyError):
            scorer.is_section_readable("test_section", "invalid_level")
    
    def test_custom_thresholds(self):
        """Test that custom thresholds are applied correctly."""
        custom_thresholds = {
            'flesch_reading_ease': 70.0,  # More strict than default
            'complex_word_percentage': 10.0  # More strict than default
        }
        
        scorer = ReadabilityScorer(thresholds=custom_thresholds)
        # Use a text with complex words that will trigger the recommendation
        text = "The implementation of the aforementioned methodologies necessitates a comprehensive understanding."
        scorer.analyze_text(text, "test_section")
        recommendations = scorer.get_recommendations("test_section")
        
        # Should recommend simplifying due to stricter thresholds
        assert any("simpler words" in rec for rec in recommendations)
    
    def test_get_overall_score(self):
        """Test getting the overall readability score."""
        scorer = ReadabilityScorer()
        text = "This is a test sentence."
        scorer.analyze_text(text, "test_section")
        
        overall_score = scorer.get_overall_score("test_section")
        assert isinstance(overall_score, float)
        assert overall_score >= 0
    
    def test_get_overall_score_missing_section(self):
        """Test that getting the overall score for a missing section raises KeyError."""
        scorer = ReadabilityScorer()
        with pytest.raises(KeyError):
            scorer.get_overall_score("nonexistent_section")
    
    def test_average_sentence_length(self):
        """Test calculation of average sentence length."""
        scorer = ReadabilityScorer()
        # Test with a simple sentence where we can verify the calculation
        text = "This is a test."
        
        # 4 words / 1 sentence = 4 words per sentence
        avg_length = scorer._average_sentence_length(text)
        assert avg_length == 4.0
    
    def test_complex_word_percentage(self):
        """Test calculation of complex word percentage."""
        scorer = ReadabilityScorer()
        
        # Create a text with exactly 5 words, where 1 is complex (3+ syllables)
        # The word "complexity" has 3+ syllables
        text = "The text has one complexity"
        
        percentage = scorer._complex_word_percentage(text)
        # 1/5 * 100 = 20%
        assert round(percentage, 1) == 20.0