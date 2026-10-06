"""Controlled modifications built from the unchanged Phase 1 models."""

from torch import nn

from rotnet.models import make_model

VARIANTS = ("learned", "identity", "random-fixed", "final-gelu", "coordinate-only", "no-affine")


class FinalGELUCore(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stages = original.stages
        self.strides = original.strides
        for stage in self.stages:
            stage.activation = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.activation(x)


class BareRotationStage(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.rotation = original.rotation
        self.activation = original.activation

    def forward(self, x):
        return self.activation(self.rotation(x))


def controlled_model(name, *, stages=8, variant="learned", representation="28x28",
                     hidden_dim=256, rank=32, threshold=0.90):
    if variant not in VARIANTS:
        raise ValueError(f"unknown component variant: {variant}")
    if representation not in ("28x28", "16x16"):
        raise ValueError("representation must be 28x28 or 16x16")
    if representation == "16x16" and hidden_dim != 256:
        raise ValueError("direct 16x16 representation requires hidden_dim=256")
    if name == "projection-only":
        model = make_model("dense", hidden_dim, rank, threshold)
        model.core = nn.Identity()
    else:
        model = make_model(name, hidden_dim, rank, threshold)
    if name in ("rotation", "adaptive-rotation"):
        if not 1 <= stages <= len(model.core.stages):
            raise ValueError("invalid stage count")
        # Construct all stages before truncating: shared prefix initialization
        # is identical at each seed for 1/2/4/8-stage learned models.
        model.core.stages = nn.ModuleList(list(model.core.stages)[:stages])
        model.core.strides = model.core.strides[:stages]
        if variant in ("identity", "random-fixed"):
            for stage in model.core.stages:
                stage.rotation.theta.requires_grad_(False)
                if variant == "identity":
                    stage.rotation.theta.data.zero_()
        elif variant == "coordinate-only":
            for stage in model.core.stages:
                stage.rotation = nn.Identity()
        elif variant == "no-affine":
            model.core.stages = nn.ModuleList([BareRotationStage(s) for s in model.core.stages])
        elif variant == "final-gelu":
            if name == "adaptive-rotation":
                raise ValueError("final-gelu control is defined only for fixed RotationNet")
            model.core = FinalGELUCore(model.core)
    elif variant != "learned" or stages != 8:
        raise ValueError("stage/component ablations apply only to rotation models")
    if representation == "16x16":
        # Discard the projection entirely, with no replacement input GELU.
        # Construction still consumes the same initialization RNG so all
        # models keep comparable classifier/core initialization.
        model.projection = nn.Identity()
        model.activation = nn.Identity()
    model.phase2_representation = representation
    model.phase2_variant = variant
    model.phase2_name = name
    return model


def arithmetic(model, mean_stages=None, final_head_only=False):
    n = model.hidden_dim
    name, variant = model.phase2_name, model.phase2_variant
    projection = 0 if model.phase2_representation == "16x16" else 2 * 784 * n
    heads, gelus = 1, n
    if name == "dense":
        core = 2 * n * n
    elif name == "low-rank":
        core = 4 * n * model.rank - model.rank
    elif name == "projection-only":
        core, gelus = 0, 0
    else:
        stages = len(model.core.stages) if mean_stages is None else mean_stages
        per_stage = (2 if variant == "coordinate-only" else 3 if variant == "no-affine" else 5) * n
        core = per_stage * stages
        gelus = n if variant == "final-gelu" else n * stages
        if name == "adaptive-rotation" and not final_head_only:
            heads = stages
    classifier = 2 * n * 10 * heads
    return {"core_multiply_add_ops_per_sample": core,
            "projection_multiply_add_ops_per_sample": projection,
            "classifier_multiply_add_ops_per_sample": classifier,
            "whole_model_multiply_add_ops_per_sample": projection + core + classifier,
            "core_gelu_coordinates_per_sample": gelus,
            "excluded": ["GELU", "sin/cos", "softmax", "threshold decisions", "memory traffic", "indexing"]}
