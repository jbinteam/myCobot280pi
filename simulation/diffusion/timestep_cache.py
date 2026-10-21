"""Scoped lookup-only fast path for the installed stock DDPM timestep schedule.

DDPM 0.40.0 searches its active timetable twice per denoising step using a
comparison and nonzero. On CUDA that lookup synchronizes the host. The policy
already visits that exact timetable in order: retain its scalar tensor views
and answer only queries for the exact currently yielded tensor object.

No scheduler equations, coefficients, random draws or tensor devices change.
Other scheduler versions/types, custom timetables and customized methods keep
the stock path. Custom float timetables can collapse to duplicate integers in
upstream validation, making a sequential successor differ from its first hit.
The caller must set a stock valid schedule before entering and must not mutate
its contents during the pass; replacing the schedule disables the fast path.
"""
import diffusers
from diffusers import DDPMScheduler
import torch

SUPPORTED_DIFFUSERS_VERSION = '0.40.0'
_STOCK_PREVIOUS = DDPMScheduler.previous_timestep
_STOCK_SET_TIMESTEPS = DDPMScheduler.set_timesteps
_MISSING = object()


def _stock_bound(method, scheduler, function):
    return getattr(method, '__self__', None) is scheduler and getattr(method, '__func__', None) is function


class PreviousTimestepCache:
    """One non-reentrant scheduler pass; enter, iterate, then always restore.

    Hits require the active scalar's Python object identity. Equal-valued
    copies, ordinary integers and out-of-order probes retain stock behavior.
    Counters describe actual wrapper use, not merely a requested feature flag.
    """
    def __init__(self, scheduler, *, enabled=True):
        self.scheduler = scheduler
        self.enabled = bool(enabled)
        self.active = False
        self.hits = self.misses = 0
        self._entered = False
        self._active_timestep = self._previous = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError('A timestep cache context cannot be entered twice')
        self._entered = True
        scheduler = self.scheduler
        self._original = scheduler.previous_timestep
        self._instance_original = vars(scheduler).get('previous_timestep', _MISSING)
        self._schedule = scheduler.timesteps
        self._steps = None
        if (self.enabled and diffusers.__version__ == SUPPORTED_DIFFUSERS_VERSION
                and type(scheduler) is DDPMScheduler
                and _stock_bound(self._original, scheduler, _STOCK_PREVIOUS)
                and _stock_bound(scheduler.set_timesteps, scheduler, _STOCK_SET_TIMESTEPS)
                and not scheduler.custom_timesteps
                and type(scheduler.num_inference_steps) is int
                and scheduler.num_inference_steps > 0
                and type(self._schedule) is torch.Tensor
                and self._schedule.ndim == 1 and len(self._schedule) == scheduler.num_inference_steps
                and self._schedule.dtype == torch.int64):
            # Tensor iteration itself uses unbind: keep the same scalar views,
            # without .item(), int(cuda_tensor), CPU copies or coefficient moves.
            self._steps = tuple(self._schedule.unbind())
            self._terminal = torch.tensor(-1)  # Exact stock terminal expression.
            scheduler.previous_timestep = self._lookup
            self.active = True
        return self

    def _lookup(self, timestep):
        if (self.active and self._active_timestep is not None
                and timestep is self._active_timestep
                and self.scheduler.timesteps is self._schedule):
            self.hits += 1
            return self._previous
        self.misses += 1
        return self._original(timestep)

    def __iter__(self):
        if not self._entered:
            raise RuntimeError('Enter the timestep cache before iterating')
        if not self.active:
            yield from self.scheduler.timesteps
            return
        try:
            for index, timestep in enumerate(self._steps):
                self._active_timestep = timestep
                self._previous = self._steps[index+1] if index+1 < len(self._steps) else self._terminal
                yield timestep
        finally:
            self._active_timestep = self._previous = None

    def __exit__(self, *exc):
        self._active_timestep = self._previous = None
        if self.active:
            if self._instance_original is _MISSING:
                del self.scheduler.previous_timestep
            else:
                self.scheduler.previous_timestep = self._instance_original
        self.active = False
        return False
