"""Compatibility patch for circuit-tracer 0.5.0 on Apple MPS, applied from our code (the library is
not edited). The lazy transcoder loaders call safetensors.safe_open(device=str(self.device)), which
is "mps:0" on MPS; safetensors accepts "mps" but rejects "mps:0". Import this module before loading
a ReplacementModel. (Also required: safetensors 0.7.0, pinned in circuit/requirements.txt; 0.8.0
rejects the tensor index the lazy decoder uses in get_slice.)"""
import circuit_tracer.transcoder.cross_layer_transcoder as _clt
import circuit_tracer.transcoder.single_layer_transcoder as _slt

_orig = _slt.safe_open


def _safe_open(path, framework="pt", device="cpu"):
    d = str(device)
    return _orig(path, framework=framework, device="mps" if d.startswith("mps") else d)


_slt.safe_open = _safe_open
_clt.safe_open = _safe_open
