from setuptools import setup, find_packages

setup(
    name="autovision",
    version="0.2.0",
    description="Car make/model/generation recognition engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "Pillow>=9.0",
    ],
    extras_require={
        "tflite": ["tflite-runtime>=2.14"],
        "onnx": ["onnxruntime>=1.16"],
        "tf": ["tensorflow>=2.14"],
        "all": ["tflite-runtime>=2.14", "onnxruntime>=1.16"],
    },
)
