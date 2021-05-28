
import setuptools

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='histomics_detect',
    version='0.0.1',
    author='Lee Cooper',
    author_email='lee.cooper@northwestern.edu',
    description='A TensorFlow 2 package for cell detection',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/CancerDataScience/histomics_detect',
    packages=setuptools.find_packages(exclude=['tests']),
    package_dir={
        'histomics_detect': 'histomics_detect',
    },
    install_requires=[
      'matplotlib',
      'numpy',
      'pandas',
      'PIL',
      'tensorflow>=2.0',
      'tensorflow_addons',
    ],
    license='Apache Software License 2.0',
    keywords='histomics_detect',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    zip_safe=False,
    python_requires='>=3.6',
)
