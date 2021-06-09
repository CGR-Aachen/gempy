from setuptools import setup, find_packages
version = '2.2.8'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy',
    version=version,
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    include_package_data=True,
    install_requires=[
        'pandas>=1.0.5',
        'Theano>=1.0.4',
        'matplotlib',
        'numpy',
        'pytest',
        'seaborn>=0.9',
        'networkx',
        'scikit-image>=0.17',
        'pyvista>=0.25',
        'pyvistaqt',
        'pyqt5',
        'iPython',
    ],
    url='https://github.com/cgre-aachen/gempy',
    license='LGPL v3',
    author='Miguel de la Varga, Alexander Zimmerman, Elisa Heim, Alexander Schaaf, Fabian Stamm, Florian Wellmann',
    author_email='varga@aices.rwth-aachen.de',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty']
)
