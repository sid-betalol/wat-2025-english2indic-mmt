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

    Focus on MAJOR issues - ignore minor stylistic differences:

    1. VISUAL CONTEXT NEEDED: Translation depends on visual information
       - Ambiguous words with multiple meanings (e.g., "dish" = food/container)
       - Gender-specific terms requiring visual verification
       - Spatial/directional terms (left/right/above/beside)
       - Physical attributes (color, size, material, quantity)
       - Object types/categories visible in image

    2. POOR TRANSLATION: Incorrect, incomplete, or unnatural
       - Mistranslation or wrong meaning (semantic error)
       - Missing key information from English
       - Severe grammatical errors making it hard to understand
       - Completely unnatural phrasing (not just stylistic preference)
       - Wrong script or excessive script mixing

    IGNORE these minor issues (mark as CORRECT):
    - Minor punctuation differences (|, ., etc.)
    - Optional articles or particles (a/the/one equivalents)
    - Stylistic word order variations (both correct)
    - Minor postposition variations if meaning is clear

    Empty captions: mark "incorrect" with "visual_context_needed"
    """

    # Inputs
    image: dspy.Image = dspy.InputField(desc="Cropped image region")
    english_caption: str = dspy.InputField(desc="Original English caption")
    target_caption: str = dspy.InputField(desc="Caption in target language to evaluate")
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

    def __init__(self, lm: dspy.LM | None = None):
        """Initialize the caption judge.

        Args:
            lm: Language model to use (if None, uses globally configured LM)
        """
        super().__init__()
        if lm:
            with dspy.context(lm=lm):
                self.judge = dspy.ChainOfThought(CaptionJudgment)
        else:
            self.judge = dspy.ChainOfThought(CaptionJudgment)
        self.lm = lm

    async def aforward(
        self,
        image: Image.Image,
        english_caption: str,
        target_caption: str,
        target_language: str,
        temp_image_path: Path | None = None,
    ) -> dspy.Prediction:
        """Judge the quality of a target language caption.

        Args:
            image: PIL Image object of the cropped region
            english_caption: Original English caption
            target_caption: Caption in target language to evaluate
            target_language: Target language name
            temp_image_path: Optional pre-saved temp file path
                (optimization to avoid recreating temp files)

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

        # Use provided temp path or create new one
        should_cleanup = False
        if temp_image_path is None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                image.save(tmp_path, format="PNG")
                should_cleanup = True
        else:
            tmp_path = temp_image_path

        try:
            if self.lm:
                with dspy.context(lm=self.lm):
                    result = await self.judge.acall(
                        image=dspy.Image(url=str(tmp_path)),
                        english_caption=english_caption,
                        target_caption=target_caption,
                        target_language=target_language,
                    )
            else:
                result = await self.judge.acall(
                    image=dspy.Image(url=str(tmp_path)),
                    english_caption=english_caption,
                    target_caption=target_caption,
                    target_language=target_language,
                )
        finally:
            # Clean up temporary file only if we created it
            if should_cleanup:
                tmp_path.unlink(missing_ok=True)

        return result
