from setuptools import setup, find_packages

setup(
    name="longppl",                 # 工具包名称
    version="0.1.3",                   # 版本号
    author="lzfang",                # 作者
    author_email="lzfang@stu.pku.edu.cn",  # 作者邮箱
    description="Calculate the longppl of long-context LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PKU-ML/LongPPL",  # 项目主页
    packages=["longppl"],          # 自动找到包含的包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',           # Python 版本要求
    install_requires=[                 # 依赖包
        "transformers>=4.44.0",
        "numpy",
        "torch",
        "datasets",
    ],
)