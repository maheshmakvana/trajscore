from setuptools import setup, find_packages

setup(
    name="trajscore",
    version="1.1.3",
    description="Production-grade agentic trajectory evaluation — score multi-step AI agent runs on goal completion, tool accuracy, step efficiency, reasoning coherence, loop detection, and faithfulness",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maheshmakvana/trajscore",
    packages=find_packages(exclude=["tests*", "venv*"]),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ],
    keywords=[
        "agent evaluation", "trajectory evaluation", "llm agent",
        "agentic ai", "ai evaluation", "tool use", "multi-step reasoning",
        "agent testing", "ai agent metrics", "goal completion",
        "step efficiency", "loop detection", "reasoning coherence",
        "answer faithfulness", "agent benchmark", "ai observability",
        "agentic benchmark", "agent trajectory", "llm testing",
        "agent quality", "production ai", "ai quality assurance",
    ],
)
