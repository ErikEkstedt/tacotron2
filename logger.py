import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import (
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    plot_gate_outputs_to_numpy,
)
from plotting_utils import get_alignment_fig, get_spectrogram_fig, get_gate_fig


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, glow=False):
        super(Tacotron2Logger, self).__init__(logdir)
        self.glow = glow

    def log_training(
        self,
        reduced_loss,
        mel_or_glow_loss,
        gate_loss,
        grad_norm,
        learning_rate,
        duration,
        iteration,
    ):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("mel_or_glow.loss", mel_or_glow_loss, iteration)
        self.add_scalar("gate.loss", gate_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(
        self, reduced_loss, mel_or_glow_loss, gate_loss, model, y, y_pred, iteration
    ):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_scalar("mel_or_glow.loss", mel_or_glow_loss, iteration)
        self.add_scalar("gate.loss", gate_loss, iteration)

        if self.glow:
            _, gate_outputs, alignments = y_pred
        else:
            _, mel_outputs, gate_outputs, alignments = y_pred
            mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        if not self.glow:
            idx = random.randint(0, alignments.size(0) - 1)
            self.add_figure(
                "alignment",
                get_alignment_fig(alignments[idx].data.cpu().numpy().T),
                iteration,
            )

            self.add_figure(
                "mel_target",
                get_spectrogram_fig(mel_targets[idx].data.cpu().squeeze().numpy()),
                iteration,
            )

            self.add_figure(
                "mel_predicted",
                get_spectrogram_fig(mel_outputs[idx].data.cpu().squeeze().numpy()),
                iteration,
            )

            self.add_figure(
                "gate",
                get_gate_fig(
                    gate_outputs=torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(),
                    gate_targets=gate_targets[idx].data.cpu().numpy(),
                ),
                iteration,
            )

    def log_inference(self, y_pred, iteration):
        mel_outputs, gate_outputs, alignments = y_pred

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_figure(
            "Inference alignment",
            get_alignment_fig(alignments.data.cpu().squeeze().numpy().T),
            iteration,
        )
        self.add_figure(
            "Inference mel_predicted",
            get_spectrogram_fig(mel_outputs.data.cpu().squeeze().numpy()),
            iteration,
        )
        self.add_figure(
            "Inference gate",
            get_gate_fig(
                gate_outputs=torch.sigmoid(gate_outputs).data.cpu().squeeze().numpy(),
                inference=True,
            ),
            iteration,
        )
