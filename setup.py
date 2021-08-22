from distutils.core import setup
from setuptools import find_packages



setup(
  name = 'wacky-rl',
  packages=find_packages(),
  version = '0.0.5',
  license='MIT',
  description = 'Create custom reinforcement learning agents with wacky-rl.',
  author = 'Maik Schürmann',
  author_email = 'maik.schuermann97@gmail.com',
  url = 'https://github.com/maik97/wacky-rl',
  download_url = 'https://github.com/maik97/wacky-rl/archive/refs/tags/v0.0.5.tar.gz',
  keywords = ['rl', 'actor_critic', 'reinforcement-learning', 'ppo', 'a2c', 'sac', 'dqn', 'keras', 'tensorflow', 'python'],
  install_requires=[
          'tensorflow',
          'tensorflow-probability',
          'gym',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.6',
  ],
)