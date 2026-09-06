"""TileLang half of the expert chain: the FFN input producers.

`producers` holds the wrappers the CUDA backend launches -- the RMS factor
kernel, the standalone K-major XFS producer and the cooperative
out-projection/residual/RMS/XFS producer that feeds the persistent FFN.
"""
