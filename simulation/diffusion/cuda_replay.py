"""Replay unchanged eager denoiser kernels with fixed CUDA input buffers.

Only the deterministic neural forward pass is captured. DDPM scheduling and
random draws stay outside the graph, preserving generator advancement. The
policy/service already serializes sampling; this buffer owner is not reentrant.
"""
import warnings
import torch


class DenoiserReplay:
    def __init__(self,model):
        self.model=model
        self.graph=None
        self.inputs=None
        self.output=None
        self.signature=None
        self.disabled=False

    def __call__(self,*args):
        signature=tuple((tuple(x.shape),x.dtype,x.device) for x in args)
        if self.disabled or any(x.device.type!='cuda' for x in args):
            return self.model(*args)
        # Keep just one allocation/capture per policy. Unusual batch shapes use
        # the original eager path instead of growing GPU memory indefinitely.
        if self.signature is not None and signature!=self.signature:
            return self.model(*args)
        if self.graph is None:
            inputs=[x.clone() for x in args]
            try:
                with torch.cuda.device(args[0].device):
                    stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):self.model(*inputs)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=self.model(*inputs)
                    graph.replay()
                    # Fail back to unchanged eager inference if this runtime
                    # cannot reproduce the initial forward pass bit for bit.
                    expected=self.model(*args)
                    if not torch.equal(output,expected):
                        self.disabled=True
                        warnings.warn('CUDA replay differs from eager denoising; using eager inference',RuntimeWarning)
                        return expected
            except RuntimeError as exc:
                # Only known unsupported-capture errors permit recovery. The
                # context managers have restored streams/capture state here;
                # synchronizing must also succeed before using eager kernels.
                # Illegal accesses, device assertions and other CUDA faults
                # propagate rather than being hidden by an automatic retry.
                unsupported=('operation not permitted when stream is capturing',
                             'cudaerrorstreamcaptureunsupported','cuda graphs are not supported')
                if any(message in str(exc).lower() for message in unsupported):
                    torch.cuda.synchronize(args[0].device)
                    self.disabled=True
                    warnings.warn('CUDA capture unsupported; using eager inference',RuntimeWarning)
                    return self.model(*args)
                raise
            self.graph=graph;self.inputs=inputs;self.output=output;self.signature=signature
        for dest,src in zip(self.inputs,args):dest.copy_(src)
        self.graph.replay()
        # Callers may retain a prediction across subsequent forwards. Never
        # expose the static output storage that the next replay overwrites.
        return self.output.clone()
