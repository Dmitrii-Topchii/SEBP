from setuptools import setup

setup(
    name="sebp",
    version="0.1.0",
    description="Sparsity-Exploiting Backpropagation (SEBP)",
    py_modules=["sebp"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "huggingface_hub>=0.20.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.41.0",
        "scipy",
        "numpy",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)