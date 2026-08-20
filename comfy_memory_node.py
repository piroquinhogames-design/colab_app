"""Node opcional de manutenção de memória que não descarrega modelos."""

from __future__ import annotations

import gc
import os


class ModelLabMemoryCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cleanup"
    CATEGORY = "ModelLab/memory"
    DESCRIPTION = "Passa a imagem sem bloquear a conclusão; limpeza CUDA é opcional."

    def cleanup(self, image):
        # Este node fica no caminho crítico antes do SaveImage. A limpeza
        # explícita do allocator pode sincronizar a T4 por minutos depois de
        # uma renderização 1024x1024; por padrão, apenas encaminhamos a imagem.
        if os.environ.get("MODELLAB_CLEANUP_CUDA", "0").strip().lower() not in {"1", "true", "yes"}:
            return (image,)
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
