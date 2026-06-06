"""
The main contribution: 
    PID-Lagrangian Consistency-Constrained Policy Optimization (PID-CCPO).

This enforces NEES consistency as a hard constraint via
PID-controlled Lagrangian relaxation rather than fragile reward shaping.

This combines:
  - SAC (Soft Actor-Critic) for the base RL objective.
  - PID-Lagrangian dual variable for the consistency constraint.
  - Cost critic for predicting constraint violations.
"""
