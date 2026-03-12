#!/usr/bin/env python3
"""
Formative 2 — System Simulation CLI

Simulates the User Identity and Product Recommendation flow:
1. Face recognition (must match known member)
2. Product recommendation (based on merged customer data)
3. Voice verification (must match approved voiceprint)
4. Display predicted product only if all steps pass

Usage:
  python scripts/run_system.py --demo-full --face-image path --voice-audio path --customer-id N
  python scripts/run_system.py --demo-unauthorized-face --face-image path
  python scripts/run_system.py --demo-unauthorized-voice --voice-audio path

Run from project root.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Formative 2 — User Identity and Product Recommendation System Simulation"
    )
    parser.add_argument(
        "--demo-full",
        action="store_true",
        help="Run full transaction: face → product → voice → display",
    )
    parser.add_argument(
        "--demo-unauthorized-face",
        action="store_true",
        help="Simulate unauthorized face attempt (access denied at face step)",
    )
    parser.add_argument(
        "--demo-unauthorized-voice",
        action="store_true",
        help="Simulate unauthorized voice attempt (access denied at voice step)",
    )
    parser.add_argument(
        "--face-image",
        type=str,
        help="Path to face image (e.g. data/images/member1/member1_neutral.jpg)",
    )
    parser.add_argument(
        "--voice-audio",
        type=str,
        help="Path to voice audio file (e.g. data/audio/samples/Preye-REC.m4a)",
    )
    parser.add_argument(
        "--customer-id",
        type=int,
        default=1,
        help="Customer ID for product recommendation (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not any([args.demo_full, args.demo_unauthorized_face, args.demo_unauthorized_voice]):
        parser.print_help()
        print("\nError: Specify at least one demo mode (--demo-full, --demo-unauthorized-face, --demo-unauthorized-voice)")
        sys.exit(1)

    print("=" * 60)
    print("Formative 2 — System Simulation")
    print("=" * 60)

    # TODO (Preye): Implement Task 4 steps
    # 1. Load face model, voice model, product model
    # 2. Implement face recognition step
    # 3. Implement product prediction step
    # 4. Implement voice verification step
    # 5. Wire full transaction flow
    # 6. Add unauthorized face attempt simulation
    # 7. Add unauthorized voice attempt simulation

    print("\n[Task 4 — Placeholder] System simulation not yet implemented.")
    print("Models and data are organized in the project root.")
    print("See formative2-task-breakdown.md for implementation steps.")
    print("=" * 60)


if __name__ == "__main__":
    main()
