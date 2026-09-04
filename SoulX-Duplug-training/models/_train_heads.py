"""Shared token-level training objectives for the LLM-based models.

Every task predicts tokens from the LLM logits and, per objective, scores a
shifted cross-entropy / token-accuracy against a different label tensor.
:class:`TokenHeadsMixin` provides those primitives, and :class:`LossHead` lets
a model declare its objectives as a registry so per-head losses and
accuracies are computed and weighted through one interface.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossHead:
    """One token-level prediction head.

    Attributes:
        name: metric name; builds the log keys ``train_<name>_loss`` /
            ``train_<name>_acc`` / ``val_acc_<name>``.
        label_key: key of the label tensor in the ``batch`` dict.
        weight_attr: loss-rate field on ``train_config``; the ``"_switched"``
            variant is used when switched loss rates are active.
    """

    name: str
    label_key: str
    weight_attr: str


class TokenHeadsMixin:
    """Token-level loss/accuracy primitives.

    The host ``LightningModule`` must define ``self.lm_vocab_size``.
    """

    def compute_accuracy(self, pad_outputs, pad_targets, ignore_label):
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return numerator.float() / denominator.float()

    def _shifted_ce(self, logits, labels):
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.lm_vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def _shifted_acc(self, preds, labels):
        return self.compute_accuracy(
            preds.detach()[:, :-1],
            labels.detach()[:, 1:],
            ignore_label=-100,
        )

    def _loss_rate(self, head, switched):
        attr = head.weight_attr + ("_switched" if switched else "")
        return getattr(self.train_config, attr)

    def _compute_heads(self, logits, preds, batch, heads):
        """Per-head losses/accuracies and the weight-summed total loss."""
        switched = bool(
            self.train_config.enable_switch_loss_rate and batch["switch_loss_rate"]
        )

        losses, accs = {}, {}
        total_loss = None
        for head in heads:
            labels = batch[head.label_key]
            loss = self._shifted_ce(logits, labels)
            weighted = self._loss_rate(head, switched) * loss
            total_loss = weighted if total_loss is None else total_loss + weighted

            losses[head.name] = loss
            accs[head.name] = self._shifted_acc(preds, labels)

        return total_loss, losses, accs
