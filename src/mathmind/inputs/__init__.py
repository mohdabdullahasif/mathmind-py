"""__init__ for inputs package."""
from mathmind.inputs.nl import translate
from mathmind.inputs.voice import normalize_transcript, voice_to_expression

__all__ = ["translate", "normalize_transcript", "voice_to_expression"]
