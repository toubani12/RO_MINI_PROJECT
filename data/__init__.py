"""
data/__init__.py
----------------
Module de génération et de chargement des instances TSP.

Expose generate_instance() comme point d'entrée principal.
"""

from .generator import generate_instance, generate_batch

__all__ = ["generate_instance", "generate_batch"]
