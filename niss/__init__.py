import logging
# Public
__all__ = ['lfa',
           'ml',
           'utils',
           'config']
__import__("pkg_resources").declare_namespace(__name__)
logging.basicConfig(format='%(message)s', level=logging.INFO)
