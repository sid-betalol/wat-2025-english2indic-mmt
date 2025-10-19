"""DSPy module for correcting captions using visual context."""

import tempfile
from pathlib import Path

import dspy
from PIL import Image


class VisualCaptionCorrectionSignature(dspy.Signature):
    """Expert translator creating natural Indian language captions.

    Generate captions by:
    1. Analyzing the IMAGE to understand visual context
    2. Using visual details to resolve ambiguities (e.g., "dish"=food/container)
    3. Creating natural captions that native speakers would use
    4. Matching English meaning while respecting target language style

    Target language lengths (be concise, not verbose):
    - Hindi: similar word count to English
    - Bengali: ~20% fewer words
    - Malayalam: ~25% fewer words
    - Odia: similar word count to English

    Note: Original caption may be wrong/missing - trust the IMAGE first.
    """

    # Inputs
    image: dspy.Image = dspy.InputField(
        desc="Cropped image region - analyze this for context"
    )
    english_caption: str = dspy.InputField(desc="English reference caption")
    target_language: str = dspy.InputField(
        desc="Target language (hindi, bengali, malayalam, odia)"
    )
    original_target_caption: str = dspy.InputField(
        desc="Original caption (may be incorrect/missing)"
    )

    # Outputs
    corrected_caption: str = dspy.OutputField(
        desc="Natural, accurate caption in target language"
    )
    explanation: str = dspy.OutputField(
        desc="What you corrected and why (1-2 sentences)"
    )


class VisualCaptionCorrector(dspy.Module):
    """DSPy module for correcting captions using visual context."""

    def __init__(self):
        """Initialize the caption corrector."""
        super().__init__()
        self.corrector = dspy.ChainOfThought(VisualCaptionCorrectionSignature)

    def forward(
        self,
        image: Image.Image,
        english_caption: str,
        target_language: str,
        original_target_caption: str = "",
    ) -> dspy.Prediction:
        """Generate corrected caption in target language.

        Args:
            image: PIL Image object of the cropped region
            english_caption: Original English caption
            target_language: Target language name
            original_target_caption: Original caption (possibly incorrect/
            missing)

        Returns:
            DSPy Prediction with corrected_caption and explanation
        """
        # Save image to temporary file and pass path
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            image.save(tmp_path, format="PNG")

        try:
            result = self.corrector(
                image=dspy.Image(url=str(tmp_path)),
                english_caption=english_caption,
                target_language=target_language,
                original_target_caption=original_target_caption or "[MISSING]",
            )
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

        return result
