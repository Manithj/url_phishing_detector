"""
Feature Extractor
=================
Extracts numerical features from URLs for ML models.
"""

import logging
import re
from typing import Dict
from urllib.parse import urlparse

logger = logging.getLogger("PhishingOrchestrator")


class FeatureExtractor:
    """
    Feature extraction class.
    Converts raw input (URL/Email) into formats required by ML and DL models.
    """

    def __init__(self):
        logger.info("FeatureExtractor initialized")

    def extract_url_features(self, url: str) -> Dict:
        """Extract numerical features from URL for ML model."""
        features = {}

        # Length features
        features['url_length'] = len(url)

        # Character counts
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_percent'] = url.count('%')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['digit_ratio'] = features['num_digits'] / max(features['url_length'], 1)

        # Special patterns
        features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['has_https'] = 1 if 'https' in url.lower() else 0
        features['has_http'] = 1 if url.lower().startswith('http://') else 0
        features['has_shortening'] = 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd', url.lower()) else 0

        # Suspicious keywords
        suspicious_words = ['login', 'signin', 'verify', 'account', 'update', 'secure',
                            'banking', 'confirm', 'password', 'credential', 'suspended']
        features['suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())

        # Domain extraction
        try:
            url_parsed = url if url.startswith('http') else 'http://' + url
            parsed = urlparse(url_parsed)
            domain = parsed.netloc
            features['domain_length'] = len(domain)
            features['path_length'] = len(parsed.path)
            features['subdomain_count'] = domain.count('.') - 1 if domain else 0
        except Exception:
            features['domain_length'] = 0
            features['path_length'] = 0
            features['subdomain_count'] = 0

        return features
