from setuptools import setup, find_packages

setup(
    name="uas_dip",
    version="0.1",
    description="Aplikasi klasifikasi jenis beras dengan deep learning",
    author="clath1a",
    author_email="alethiaclarita1@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'tensorflow',
        'tqdm',
        'flask',  # Untuk bagian web app
        'opencv-python'  # Diasumsikan cx2 adalah OpenCV
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        '': ['static/css/*', 'static/uploads/*', 'templates/*']
    }
)