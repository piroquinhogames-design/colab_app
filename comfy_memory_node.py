"""Node opcional de manutenção de memória que não descarrega modelos."""

from __future__ import annotations

import gc


class ModelLabMemoryCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cleanup"
    CATEGORY = "ModelLab/memory"
    DESCRIPTION = "Libera temporários ociosos; não descarrega modelos da GPU."

    def cleanup(self, image):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        return (image,)


NODE_CLASS_MAPPINGS = {"ModelLabMemoryCleanup": ModelLabMemoryCleanup}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelLabMemoryCleanup": "ModelLab memory cleanup (resident models)"}
