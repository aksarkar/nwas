import setuptools

setuptools.setup(
    name='nwas',
    description='Intermediate phenotype-wide association studies',
    version='0.1',
    url='https://www.github.com/aksarkar/nwas',
    author='Abhishek Sarkar',
    author_email='aksarkar@uchicago.edu',
    license='MIT',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'tensorflow', 'edward'],
)
