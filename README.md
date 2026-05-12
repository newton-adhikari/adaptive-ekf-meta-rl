# CC-MetaEKF: Filter-Aware Meta-RL for Online EKF Adaptation

> **Spectral-Context Meta-Reinforcement Learning with Empirical Consistency Constraints for Online EKF Noise Adaptation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)

## Abstract

The Extended Kalman Filter (EKF) underpins state estimation across robotics from warehouse AGVs to planetary rovers yet a single misspecified noise covariance can silently degrade localization until the robot is lost. When the environment changes at runtime (terrain transitions, sensor degradation, weather shifts), hand-tuned noise parameters become liabilities. Classical adaptive filters (Sage-Husa, innovation-based) assume the noise is stationary, precisely the assumption that fails in practice. End-to-end learned filters (KalmanNet) can handle non-stationarity but sacrifice the interpretability and consistency analysis tools that make the EKF trustworthy. Naive application of meta-RL to EKF tuning ignores the rich statistical structure of the filtering problem.

We propose **CC-MetaEKF**, a filter-aware meta-RL framework for online EKF noise adaptation that preserves the classical EKF structure while learning to adapt its noise parameters. Our three contributions are: (1) a **Short-Time Spectral Innovation Encoder (ST-SIE)** that extracts noise-diagnostic features from the STFT of the innovation sequence under a local stationarity assumption, providing a principled inductive bias that generic encoders lack; (2) **PID-Lagrangian Consistency-Constrained Policy Optimization (PID-CCPO)**, which empirically enforces NEES consistency as a hard constraint via PID-controlled Lagrangian relaxation rather than fragile reward shaping; and (3) a **calibrated sim-to-real pipeline** where the training task distribution is anchored to real sensor noise models collected on the target platform.

## What This Project Is and Isn't

**What it is**: A systems/methods paper that introduces filter-specific inductive biases (ST-SIE) and a practical constrained optimization approach (PID-CCPO) to the problem of online EKF noise adaptation via meta-RL. The contributions are architectural and algorithmic, validated empirically.

**What it isn't**: We do not provide formal convergence guarantees[as this is still in training] for the constrained optimization (extending Tessler et al. 2019 to deep RL with learned encoders would require substantially more theoretical work). We are honest about this limitation.