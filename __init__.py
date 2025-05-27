from .nodes import LoadAniSoraModel, AniSoraPrompt, AniSora, SaveAniSora

NODE_CLASS_MAPPINGS = {
    "LoadAniSoraModel": LoadAniSoraModel,
    "AniSoraPrompt": AniSoraPrompt,
    "AniSora": AniSora,
    "SaveAniSora": SaveAniSora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAniSoraModel": "Load AniSora Model",
    "AniSoraPrompt": "AniSora Prompt",
    "AniSora": "AniSora",
    "SaveAniSora": "Save AniSora",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
