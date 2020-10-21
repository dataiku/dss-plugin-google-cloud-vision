# -*- coding: utf-8 -*-
from language_dict import SUPPORTED_LANGUAGES


def do(payload, config, plugin_config, inputs):
    """Compute Language SELECT choices for the front-end of the Image and Document Text Detection recipes"""
    language_choices = sorted(
        [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES.items()], key=lambda x: x.get("label")
    )
    language_choices.insert(0, {"label": "Auto-detect", "value": ""})
    return {"choices": language_choices}
