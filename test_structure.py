"""
Basic tests for Photo Organizer components.
These tests verify the structure and basic functionality without requiring TensorFlow.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestClassifierStructure(unittest.TestCase):
    """Test the classifier module structure."""
    
    def test_classifier_import(self):
        """Test that classifier module can be imported."""
        try:
            import classifier
            self.assertTrue(hasattr(classifier, 'HomeworkImageClassifier'))
        except ImportError as e:
            # Expected if TensorFlow not installed
            self.assertIn('tensorflow', str(e).lower())
    
    def test_classifier_has_required_methods(self):
        """Test that classifier class has required methods."""
        # Mock all dependencies to avoid import errors
        with patch.dict('sys.modules', {
            'tensorflow': MagicMock(),
            'tensorflow.keras': MagicMock(),
            'tensorflow.keras.applications': MagicMock(),
            'tensorflow.keras.applications.efficientnet': MagicMock(),
            'tensorflow.keras.preprocessing': MagicMock(),
            'tensorflow.keras.preprocessing.image': MagicMock(),
            'numpy': MagicMock(),
            'PIL': MagicMock(),
            'PIL.Image': MagicMock(),
        }):
            import classifier
            
            # Check class exists
            self.assertTrue(hasattr(classifier, 'HomeworkImageClassifier'))
            
            # Check methods exist
            methods = ['load_and_preprocess_image', 'is_homework_related', 'classify_batch']
            for method in methods:
                self.assertTrue(
                    hasattr(classifier.HomeworkImageClassifier, method),
                    f"Missing method: {method}"
                )


class TestOrganizerStructure(unittest.TestCase):
    """Test the organizer module structure."""
    
    def test_organizer_import(self):
        """Test that organizer module structure is correct."""
        # Mock the classifier to avoid TensorFlow dependency
        mock_classifier = MagicMock()
        with patch.dict('sys.modules', {
            'classifier': mock_classifier,
            'tensorflow': MagicMock(),
        }):
            import organizer
            self.assertTrue(hasattr(organizer, 'PhotoOrganizer'))
    
    def test_organizer_initialization(self):
        """Test that PhotoOrganizer can be initialized."""
        mock_classifier = MagicMock()
        with patch.dict('sys.modules', {
            'classifier': mock_classifier,
        }):
            import organizer
            
            # Test initialization with default params
            org = organizer.PhotoOrganizer(source_dir='test_dir')
            self.assertEqual(str(org.source_dir), 'test_dir')
            self.assertEqual(str(org.homework_dir), 'school_photos')
    
    def test_organizer_custom_homework_dir(self):
        """Test PhotoOrganizer with custom homework directory."""
        mock_classifier = MagicMock()
        with patch.dict('sys.modules', {
            'classifier': mock_classifier,
        }):
            import organizer
            
            org = organizer.PhotoOrganizer(
                source_dir='test_dir',
                homework_dir='custom_folder'
            )
            self.assertEqual(str(org.homework_dir), 'custom_folder')
    
    def test_supported_image_extensions(self):
        """Test that organizer recognizes common image extensions."""
        mock_classifier = MagicMock()
        with patch.dict('sys.modules', {
            'classifier': mock_classifier,
        }):
            import organizer
            
            org = organizer.PhotoOrganizer(source_dir='test')
            
            expected_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            self.assertEqual(org.image_extensions, expected_extensions)


class TestConfigStructure(unittest.TestCase):
    """Test the configuration module."""
    
    def test_config_import(self):
        """Test that config module exists and has required settings."""
        import config
        
        # Check required config variables exist
        required_vars = [
            'CONFIDENCE_THRESHOLD',
            'HOMEWORK_KEYWORDS',
            'IMAGE_SIZE',
            'SUPPORTED_FORMATS',
            'DEFAULT_HOMEWORK_FOLDER'
        ]
        
        for var in required_vars:
            self.assertTrue(
                hasattr(config, var),
                f"Missing config variable: {var}"
            )
    
    def test_config_values(self):
        """Test that config values are reasonable."""
        import config
        
        # Test confidence threshold is between 0 and 1
        self.assertGreater(config.CONFIDENCE_THRESHOLD, 0)
        self.assertLess(config.CONFIDENCE_THRESHOLD, 1)
        
        # Test homework keywords is a list
        self.assertIsInstance(config.HOMEWORK_KEYWORDS, list)
        self.assertGreater(len(config.HOMEWORK_KEYWORDS), 0)
        
        # Test image size is a tuple
        self.assertIsInstance(config.IMAGE_SIZE, tuple)
        self.assertEqual(len(config.IMAGE_SIZE), 2)


class TestProjectStructure(unittest.TestCase):
    """Test overall project structure."""
    
    def test_required_files_exist(self):
        """Test that all required files exist."""
        base_dir = Path(__file__).parent
        
        required_files = [
            'classifier.py',
            'organizer.py',
            'config.py',
            'requirements.txt',
            'README.md',
            'setup.py',
            '.gitignore'
        ]
        
        for filename in required_files:
            file_path = base_dir / filename
            self.assertTrue(
                file_path.exists(),
                f"Missing required file: {filename}"
            )
    
    def test_readme_has_content(self):
        """Test that README has meaningful content."""
        base_dir = Path(__file__).parent
        readme_path = base_dir / 'README.md'
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        self.assertIn('Photo Organizer', content)
        self.assertIn('Installation', content)
        self.assertIn('Usage', content)
        self.assertGreater(len(content), 500, "README seems too short")
    
    def test_requirements_txt(self):
        """Test that requirements.txt has necessary dependencies."""
        base_dir = Path(__file__).parent
        req_path = base_dir / 'requirements.txt'
        
        with open(req_path, 'r') as f:
            content = f.read().lower()
        
        # Check for required packages
        self.assertIn('tensorflow', content)
        self.assertIn('pillow', content)
        self.assertIn('numpy', content)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestClassifierStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestOrganizerStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
