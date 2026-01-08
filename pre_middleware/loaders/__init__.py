"""
pre_middleware.loaders - Data Format Loaders for Raw Streams

Loaders for various raw data stream formats that need trigger detection:
- MatLoader: MATLAB .mat files (scipy.io.loadmat)
- BinaryLoader: Raw binary streams (future)
- HDF5Loader: HDF5 format files (future)

Usage:
    from pre_middleware.loaders import MatLoader

    loader = MatLoader('IEEE Data/dataset.mat')
    data = loader.load()

    print(data['signal'].shape)
    print(data['sample_rate'])
"""

from .mat_loader import MatLoader, load_mat_file

__all__ = [
    'MatLoader',
    'load_mat_file',
]
