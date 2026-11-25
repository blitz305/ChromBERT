"""Collect CPU and CUDA profiles from a bounded sample of benchmark batches."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch


class BenchmarkProfiler:
    def __init__(self, torch, output, task):
        self.api = torch.profiler
        self.cuda_device = torch.autograd.DeviceType.CUDA
        self.output = output
        self.task = task.upper()
        self.prefix = self.task + "/"
        self.results = {}

    def annotate(self, name):
        return self.api.record_function(self.prefix + name)

    def mark_forward(self, module, name, inputs=None):
        original = module.forward

        @wraps(original)
        def forward(*args, **kwargs):
            with self.annotate(name):
                if inputs is not None:
                    x = args[0]
                    inputs[(str(x.dtype), tuple(x.shape))] += 1
                return original(*args, **kwargs)

        return patch.object(module, "forward", forward)

    @contextmanager
    def capture(self, stage, model, first, last):
        print(f"Profiling {self.task} {stage}, batches {first}–{last}", flush=True)
        gelu_inputs = Counter()
        encoder = model.get_pretrain()
        with ExitStack() as stack:
            stack.enter_context(self.mark_forward(encoder, "encoder"))
            if hasattr(model, "dnabert2"):
                stack.enter_context(self.mark_forward(model.dnabert2, "dnabert2"))
                stack.enter_context(self.mark_forward(model.head_output, "fusion_head"))
            else:
                stack.enter_context(self.mark_forward(model.ft_header, "prediction_head"))
            if hasattr(model, "pool_flank_window"):
                stack.enter_context(self.mark_forward(model.pool_flank_window, "window_encoding_and_pooling"))
            if hasattr(model, "gather_emb"):
                stack.enter_context(self.mark_forward(model.gather_emb, "prompt_pooling"))
            for block in encoder.transformer_blocks:
                ffn = block.feed_forward
                stack.enter_context(self.mark_forward(block.attention, "attention"))
                stack.enter_context(self.mark_forward(ffn, "feed_forward"))
                stack.enter_context(self.mark_forward(ffn.w_1, "ffn_linear_in"))
                stack.enter_context(self.mark_forward(ffn.activation, "ffn_gelu", gelu_inputs))
                stack.enter_context(self.mark_forward(ffn.w_2, "ffn_linear_out"))
            profile = stack.enter_context(self.api.profile(
                activities=[self.api.ProfilerActivity.CPU, self.api.ProfilerActivity.CUDA],
                profile_memory=True,
            ))
            yield

        cuda_events = sum(event.device_type == self.cuda_device and event.name != "[memory]"
                          for event in profile.events())
        if not cuda_events:
            raise RuntimeError(f"{self.task} {stage} profiler collected no CUDA events.")
        operators = profile.key_averages()
        gpu_time = "device_time_total" if hasattr(operators[0], "device_time_total") else "cuda_time_total"
        sections = {
            op.key[len(self.prefix):]: {
                "calls": op.count, "cpu_total_ms": op.cpu_time_total / 1000,
                "cuda_total_ms": getattr(op, gpu_time) / 1000,
            }
            for op in operators if op.key.startswith(self.prefix)
        }
        if not gelu_inputs or sections["ffn_gelu"]["calls"] != sections["feed_forward"]["calls"]:
            raise RuntimeError(f"{self.task} {stage} profiler did not mark every FFN GELU call.")
        if sections["ffn_gelu"]["cuda_total_ms"] <= 0:
            raise RuntimeError(f"{self.task} {stage} profiler did not attribute CUDA time to GELU.")
        profile.export_chrome_trace(str(self.output / f"{stage}_trace.json"))
        report = (
            f"Captured batches (one-based): {first} to {last}\nCUDA events: {cuda_events}\n\n"
            "Model and data sections (inclusive times, nested sections overlap)\n"
            f"{'Section':<30} {'Calls':>6} {'CPU total ms':>14} {'CUDA total ms':>14}\n"
        )
        for name, section in sections.items():
            report += (f"{name:<30} {section['calls']:>6} {section['cpu_total_ms']:>14.3f} "
                       f"{section['cuda_total_ms']:>14.3f}\n")
        report += "\nTop operators by self CUDA time\n" + operators.table(sort_by=f"self_{gpu_time}", row_limit=25)
        report += "\nTop operators by self CPU time\n" + operators.table(sort_by="self_cpu_time_total", row_limit=25)
        (self.output / f"{stage}_operators.txt").write_text(report, encoding="utf-8")
        self.results[stage] = {
            "first_batch": first, "last_batch": last, "cuda_events": cuda_events,
            "sections": sections,
            "gelu_inputs": [{"dtype": dtype, "shape": list(shape), "calls": calls}
                            for (dtype, shape), calls in gelu_inputs.items()],
        }

    @contextmanager
    def training_callback(self, pl, model, *, stop_after_capture=False):
        capture = self.capture
        with ExitStack() as stack:
            class CaptureBatches(pl.Callback):
                def on_train_epoch_start(self, trainer, pl_module):
                    accumulation = trainer.accumulate_grad_batches
                    window_end = accumulation if stop_after_capture else (4 // accumulation + 1) * accumulation
                    self.last = min(int(trainer.num_training_batches), window_end)
                    self.first = max(1, self.last - min(2, accumulation) + 1)
                    # End at an optimizer update without spanning a validation boundary.
                    interval = trainer.val_check_batch
                    if interval < self.last:
                        previous_validation = ((self.last - 1) // interval) * interval
                        self.first = max(self.first, int(previous_validation) + 1)

                def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                    if trainer.current_epoch == 0 and batch_idx + 1 == self.first:
                        stack.enter_context(capture("training", model, self.first, self.last))

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    if trainer.current_epoch == 0 and batch_idx + 1 == self.last:
                        stack.close()
                        if stop_after_capture:
                            trainer.should_stop = True

            yield CaptureBatches()

    def inference_batches(self, model, batches, batch_count, *, stop_after_capture=False):
        last = min(8, batch_count)
        first = max(1, last - 3)
        batches = iter(batches)
        yield from self.capture_batches("inference", model, batches, first, last)
        if not stop_after_capture:
            yield from batches

    def capture_batches(self, stage, model, batches, first, last):
        for _ in range(first - 1):
            yield next(batches)
        with self.capture(stage, model, first, last):
            for _ in range(first, last + 1):
                with self.annotate("data_wait"):
                    batch = next(batches)
                yield batch
