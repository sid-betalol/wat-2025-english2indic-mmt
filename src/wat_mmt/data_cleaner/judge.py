"""DSPy module for judging caption quality."""

import tempfile
from pathlib import Path

import dspy
from PIL import Image


class CaptionJudgment(dspy.Signature):
    """You are an expert multilingual translator evaluating Indian language
    captions.

    Determine if the target language caption correctly represents what's
    shown in the image and accurately conveys the English caption meaning.

    Common issues (Be STRICT - flag even slight inconsistencies):

    1. VISUAL CONTEXT NEEDED: Translation depends on visual information
       - Ambiguous words with multiple meanings (e.g., "dish" = food/container)
       - Gender-specific terms requiring visual verification
       - Spatial/directional terms (left/right/above/beside)
       - Physical attributes (color, size, material, quantity)
       - Object types/categories visible in image

    2. POOR TRANSLATION: Incorrect, incomplete, or unnatural
       - Mistranslation or wrong meaning
       - Missing/incomplete information
       - Grammatical errors (gender, case, verb forms, postpositions)
       - Unnatural phrasing (technically correct but awkward)
       - Script errors (wrong script, mixing scripts)
       - Cultural/contextual mistranslation

    Empty captions: mark "incorrect" with "visual_context_needed"
    """

    # Inputs
    image: dspy.Image = dspy.InputField(desc="Cropped image region")
    english_caption: str = dspy.InputField(desc="Original English caption")
    target_caption: str = dspy.InputField(
        desc="Caption in target language to evaluate"
    )
    target_language: str = dspy.InputField(
        desc="Target language (hindi, bengali, malayalam, odia)"
    )

    # Outputs
    status: str = dspy.OutputField(desc="'correct' or 'incorrect'")
    reason: str = dspy.OutputField(
        desc="'visual_context_needed', 'poor_translation', or 'none'"
    )
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    explanation: str = dspy.OutputField(
        desc="Brief explanation citing the specific issue (1-2 sentences)"
    )


class CaptionJudge(dspy.Module):
    """DSPy module for judging caption quality with visual context."""

    def __init__(self):
        """Initialize the caption judge."""
        super().__init__()
        self.judge = dspy.ChainOfThought(CaptionJudgment)

    def forward(
        self,
        image: Image.Image,
        english_caption: str,
        target_caption: str,
        target_language: str,
    ) -> dspy.Prediction:
        """Judge the quality of a target language caption.

        Args:
            image: PIL Image object of the cropped region
            english_caption: Original English caption
            target_caption: Caption in target language to evaluate
            target_language: Target language name

        Returns:
            DSPy Prediction with status, reason, confidence, and explanation
        """
        # Check if caption is missing or empty
        is_missing = (
            not target_caption
            or target_caption.strip() == ""
            or target_caption.strip().lower() in ["nan", "none", "null"]
        )

        if is_missing:
            # Directly return for missing captions
            return dspy.Prediction(
                status="incorrect",
                reason="visual_context_needed",
                confidence=1.0,
                explanation=(
                    "Caption is missing or empty. "
                    "Visual context needed to generate caption."
                ),
            )

        # Save image to temporary file and pass path
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            image.save(tmp_path, format="PNG")

        try:
            result = self.judge(
                image=dspy.Image(url=str(tmp_path)),
                english_caption=english_caption,
                target_caption=target_caption,
                target_language=target_language,
            )
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

        return result
