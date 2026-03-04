from src.KBLN import (
    ComplEx_KBLN,
    DeCaL_KBLN,
    DistMult_KBLN,
    DualE_KBLN,
    Keci_KBLN,
    OMult_KBLN,
    QMult_KBLN,
    TransE_KBLN,
)
from src.literalE import (
    ComplEx_LiteralE,
    DeCaL_LiteralE,
    DistMult_LiteralE,
    DualE_LiteralE,
    Keci_LiteralE,
    OMult_LiteralE,
    QMult_LiteralE,
    TransE_LiteralE,
)


LITERALE_MODEL_MAP = {
    "ComplEx": ComplEx_LiteralE,
    "DeCaL": DeCaL_LiteralE,
    "DistMult": DistMult_LiteralE,
    "DualE": DualE_LiteralE,
    "Keci": Keci_LiteralE,
    "OMult": OMult_LiteralE,
    "QMult": QMult_LiteralE,
    "TransE": TransE_LiteralE,
}


KBLN_MODEL_MAP = {
    "ComplEx": ComplEx_KBLN,
    "DeCaL": DeCaL_KBLN,
    "DistMult": DistMult_KBLN,
    "DualE": DualE_KBLN,
    "Keci": Keci_KBLN,
    "OMult": OMult_KBLN,
    "QMult": QMult_KBLN,
    "TransE": TransE_KBLN,
}


def resolve_mode_model_class(model_name: str, mode: str):
    if mode == "literalE":
        model_map = LITERALE_MODEL_MAP
        mode_name = "LiteralE"
    elif mode == "kbln":
        model_map = KBLN_MODEL_MAP
        mode_name = "KBLN"
    else:
        raise ValueError(f"Unknown model resolution mode: {mode}")

    try:
        return model_map[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(model_map))
        raise ValueError(
            f"{mode_name} subclass model is not implemented for {model_name}. Supported models: {supported}"
        ) from exc
