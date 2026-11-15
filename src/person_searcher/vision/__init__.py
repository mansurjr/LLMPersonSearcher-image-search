"""Computer-vision detection adapters."""

from .yolo import YOLOPersonDetector

from .reid import ColorHistogramEmbedder, cosine_similarity, rank_similar_crops

__all__ = ["YOLOPersonDetector", "ColorHistogramEmbedder", "cosine_similarity", "rank_similar_crops"]
