import pathlib
from pip.req import parse_requirements
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

reqs = [str(ir.req) for ir in parse_requirements(here / 'requirements.txt')]

setup(
    name='climatereconstructionAI',
    version='0.1.0',
    description='A climate data infilling software using partial convolutions',
    long_description=long_description,
    url='https://github.com/FREVA-CLINT/climatereconstructionAI',
    # author='Naoto Inoue, Christopher Kadow, Stephan Seitz, Johannes Meuer, Étienne Plésiat'.
    author='Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum)',
    # author_email='author@example.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='climate, artificial intelligence, infilling, missing values, partial convolution, NetCDF',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7, <4',
    include_package_data=True,
    install_requires=reqs,
)
