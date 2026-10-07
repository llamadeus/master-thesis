"""Architecture-neutral named-head objective reducers."""

from __future__ import annotations

import torch

from ..specs import (AuxiliaryNextLatentObjectiveSpec,
                     DistillationObjectiveSpec, LatentDynamicsObjectiveSpec,
                     NextLatentObjectiveSpec, OutcomeObjectiveSpec)


def named_objective(predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], spec):
    loss_fn = torch.nn.functional.mse_loss if spec.loss == "mse" else torch.nn.functional.l1_loss
    if isinstance(spec, OutcomeObjectiveSpec):
        weights = dict(spec.head_weights) or {name: 1.0 for name in predictions}
        missing = set(weights) - set(predictions) | set(weights) - set(targets)
        if missing: raise ValueError(f"objective references unavailable heads: {sorted(missing)}")
        components = {f"outcome/{name}": loss_fn(predictions[name], targets[name]) for name in weights}
        combined = sum(weights[name] * components[f"outcome/{name}"] for name in weights)
        return combined, {"outcome_loss": combined, **components}
    raise TypeError("named_objective requires OutcomeObjectiveSpec")


def distillation_objective(student_predictions, outcome_targets, teacher_predictions,
                           spec: DistillationObjectiveSpec):
    if spec.student_head not in student_predictions or spec.student_head not in outcome_targets:
        raise ValueError(f"destination head {spec.student_head!r} is unavailable")
    if spec.teacher_head not in teacher_predictions:
        raise ValueError(f"source head {spec.teacher_head!r} is unavailable")
    student = student_predictions[spec.student_head]
    teacher = teacher_predictions[spec.teacher_head].detach()
    if student.shape != teacher.shape:
        raise ValueError("distillation source and destination head shapes differ")
    fn = torch.nn.functional.mse_loss if spec.loss == "mse" else torch.nn.functional.l1_loss
    outcome = fn(student, outcome_targets[spec.student_head]); distilled = fn(student, teacher)
    combined = spec.outcome_weight * outcome + spec.teacher_weight * distilled
    return combined, {"combined_loss": combined, "outcome_loss": outcome, "distillation_loss": distilled}


def latent_dynamics_objective(predictions, outcome_targets, predicted_states,
                              target_states, spec: LatentDynamicsObjectiveSpec):
    if "chlorine" not in predictions or "chlorine" not in outcome_targets:
        raise ValueError("latent-dynamics objective requires chlorine predictions and targets")
    if predicted_states.shape != target_states.shape:
        raise ValueError("predicted and target latent-state shapes differ")
    chlorine = torch.nn.functional.mse_loss(predictions["chlorine"], outcome_targets["chlorine"])
    latent = torch.nn.functional.mse_loss(predicted_states, target_states.detach())
    combined = spec.chlorine_weight * chlorine + spec.latent_weight * latent
    return combined, {
        "combined_loss": combined,
        "outcome_loss": chlorine,
        "chlorine_loss": chlorine,
        "latent_loss": latent,
    }


def next_latent_objective(current_states, predicted_states, target_states,
                          spec: NextLatentObjectiveSpec | AuxiliaryNextLatentObjectiveSpec):
    if current_states.shape != predicted_states.shape or predicted_states.shape != target_states.shape:
        raise ValueError("current, predicted, and target latent-state shapes differ")
    if current_states.ndim != 2:
        raise ValueError("next-latent states must be batch-first rank-two tensors")
    next_latent_mse = torch.nn.functional.mse_loss(
        predicted_states, target_states.detach()
    )
    latent_std = torch.sqrt(
        current_states.var(dim=0, unbiased=False) + spec.variance_epsilon
    )
    if spec.variance_regularizer == "two-sided":
        variance = torch.square(latent_std - spec.variance_target_std).mean()
    else:
        variance = torch.relu(spec.variance_target_std - latent_std).mean()
    combined = next_latent_mse + spec.variance_weight * variance
    return combined, {
        "combined_loss": combined,
        "outcome_loss": next_latent_mse,
        "next_latent_mse": next_latent_mse,
        "variance_loss": variance,
        "mean_latent_std": latent_std.mean(),
    }


def auxiliary_next_latent_objective(
    current_states,
    predicted_states,
    target_states,
    predictions,
    outcome_targets,
    spec: AuxiliaryNextLatentObjectiveSpec,
):
    if "chlorine" not in predictions or "chlorine" not in outcome_targets:
        raise ValueError(
            "auxiliary next-latent objective requires chlorine predictions and targets"
        )
    latent, latent_components = next_latent_objective(
        current_states, predicted_states, target_states, spec
    )
    loss_fn = (
        torch.nn.functional.mse_loss
        if spec.loss == "mse"
        else torch.nn.functional.l1_loss
    )
    chlorine = loss_fn(predictions["chlorine"], outcome_targets["chlorine"])
    combined = latent + spec.chlorine_weight * chlorine
    return combined, {
        "combined_loss": combined,
        "outcome_loss": chlorine,
        "chlorine_loss": chlorine,
        "latent_loss": latent,
        "next_latent_mse": latent_components["next_latent_mse"],
        "variance_loss": latent_components["variance_loss"],
        "mean_latent_std": latent_components["mean_latent_std"],
    }
