from setuptools import setup

# read the contents of README
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
  name = 'cellvgae',
  packages = ['cellvgae'],
  version = '0.0.1b0',
  license='MIT',
  description = 'CellVGAE uses the connectivity between cells (such as k-nearest neighbour graphs) with gene expression values as node features to learn high-quality cell representations in a lower-dimensional space',
  author = 'David Buterez',
  author_email = 'david.buterez@gmail.com',
  url = 'https://github.com/davidbuterez/CellVGAE',
  keywords = ['scrnaseq', 'graph', 'gnn', 'dimensionality', 'reduction', 'neural'],
  install_requires=[
        "torch>=1.6.0",
        "umap_learn>=0.5.1",
        "hdbscan>=0.8.27",
        "faiss-cpu>=1.7.0",
        "seaborn>=0.11.1",
        "matplotlib>=3.3.4",
        "scanpy>=1.7.2",
        "anndata>=0.7.5",
        "tqdm>=4.61.2",
        "termcolor>=1.1.0",
        "numpy>=1.19.5",
        "pandas>=1.2.4",
        "torch_geometric>=1.7.0",
        "scikit_learn>=0.24.2",
        "umap>=0.1.1",
        "torch_sparse>=0.6.12",
        "torch_scatter>=2.0.8"
    ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
  long_description=long_description,
  long_description_content_type='text/markdown'
)