"""Example script demonstrating inference with finetuned IndicTrans2."""

from pathlib import Path

from src.wat_mmt.finetuning import IndicTrans2Translator


def main():
    """Run inference examples."""
    # Initialize translator
    print("Loading model...")
    translator = IndicTrans2Translator(
        model_path=Path("models/indictrans2-lora-corrected"),
        is_lora=True,
    )

    # Example 1: Single translation
    print("\n" + "=" * 80)
    print("Example 1: Single Translation")
    print("=" * 80)

    text = "white block on tower"
    for lang in ["hindi", "bengali", "malayalam", "odia"]:
        translation = translator.translate(text, target_lang=lang)
        print(f"\n{lang.capitalize()}: {translation}")

    # Example 2: Batch translation
    print("\n" + "=" * 80)
    print("Example 2: Batch Translation")
    print("=" * 80)

    texts = [
        "the head of a girl",
        "man on a paddle board",
        "fruit on the plate",
        "this is a dog",
    ]

    translations = translator.translate(texts, target_lang="hindi")
    print(f"\nTranslating {len(texts)} texts to Hindi:")
    for text, trans in zip(texts, translations):
        print(f"  {text} → {trans}")

    # Example 3: Different languages
    print("\n" + "=" * 80)
    print("Example 3: Same text in all languages")
    print("=" * 80)

    text = "the sky is gray"
    for lang in ["hindi", "bengali", "malayalam", "odia"]:
        translation = translator.translate(text, target_lang=lang)
        print(f"\n{lang.capitalize()}: {translation}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
