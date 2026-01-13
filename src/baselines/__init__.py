"""Baseline adapters for AutoGen-MAC and LlamaIndex-Workflows.

These modules provide adapters to run the two baselines using the
workspace's modality processors (`TextHandler`, `VoiceHandler`, `ImageHandler`)
with identical inputs/outputs as our dispatcher, ensuring fair comparison.

Both adapters attempt to import their respective upstream libraries. If the
libraries are unavailable, they fall back to internally orchestrated flows
that mimic the described behavior using our processors.
"""
