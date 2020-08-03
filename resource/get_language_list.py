# -*- coding: utf-8 -*-
from language_dict import SUPPORTED_LANGUAGES


def do(payload, config, plugin_config, inputs):
    language_choices = SUPPORTED_LANGUAGES
    language_choices.insert(0, {"label": "Auto-detect", "value": ""})
    return {"choices": language_choices}
