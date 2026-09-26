'''Independent, randomly initialized two-scale YOLO detector.'''

__all__ = ['TinyDetector']


def __getattr__(name):
    if name == 'TinyDetector':
        from .model import TinyDetector
        return TinyDetector
    raise AttributeError(name)
