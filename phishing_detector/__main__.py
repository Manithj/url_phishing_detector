"""
Phishing Detector Entry Point
=============================
Run with: uv run python -m phishing_detector
"""

import sys
from pathlib import Path

# Add parent directory to path for direct execution
if __package__ is None or __package__ == '':
    # Running directly (uv run phishing_detector)
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from phishing_detector import PhishingOrchestrator
else:
    # Running as module (python -m phishing_detector)
    from . import PhishingOrchestrator


def main():
    """Demonstrate the PhishingOrchestrator."""
    print("\n" + "=" * 60)
    print("PHISHING DETECTION ORCHESTRATOR")
    print("=" * 60)

    # Test URLs
    test_urls = [
        "www.google.com",
        "https://github.com/user/repo",
        "paypal.com.login-verify.suspicious-site.xyz/secure/update",
        "http://192.168.1.1:8080/admin/login.php?redirect=bank",
        "bit.ly/3xYz123",
    ]

    try:
        # Initialize orchestrator
        orchestrator = PhishingOrchestrator()

        print("\n" + "-" * 60)
        print("CLASSIFICATION RESULTS")
        print("-" * 60)

        for url in test_urls:
            result = orchestrator.classify(url)

            print(f"\nURL: {url}")
            print(f"  Final Verdict: {result.final_verdict}")
            print(f"  Confidence: {result.confidence:.1%}")
            
            ml_conf = f"{result.ml_result.confidence:.1%}" if result.ml_result else "N/A"
            ml_pred = result.ml_result.prediction if result.ml_result else "N/A"
            print(f"  ML: {ml_pred} ({ml_conf})")
            
            dl_conf = f"{result.dl_result.confidence:.1%}" if result.dl_result else "N/A"
            dl_pred = result.dl_result.prediction if result.dl_result else "N/A"
            print(f"  DL: {dl_pred} ({dl_conf})")
            
            genai_conf = f"{result.genai_result.confidence:.1%}" if result.genai_result else "N/A"
            genai_pred = result.genai_result.prediction if result.genai_result else "N/A"
            print(f"  GenAI: {genai_pred} ({genai_conf})")

            if result.failover_used:
                print("  [FAILOVER MODE USED]")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
