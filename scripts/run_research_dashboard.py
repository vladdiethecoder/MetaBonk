#!/usr/bin/env python3
"""Research Command Center runner.

Runs the "Nerd Tab" RCC by polling the orchestrator and learner services.
Demo/simulated telemetry is intentionally not supported.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from src.broadcast.research_dashboard import (
        ResearchDashboard,
        MetricsCollector,
    )

    parser = argparse.ArgumentParser(description="MetaBonk Research Command Center")
    parser.add_argument("--orch-url", default="http://localhost:8040", help="Orchestrator base URL")
    parser.add_argument("--learner-url", default="http://localhost:8061", help="Learner base URL")
    parser.add_argument("--policy-name", default=None, help="Policy to focus on (optional)")
    parser.add_argument("--baselines", default=None, help="Path to baselines JSON/YAML (optional)")
    parser.add_argument("--rerun", action="store_true", help="Log RCC scalars to Rerun if installed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Metabonk Research Command Center")
    print("=" * 60)
    
    # Create dashboard
    dashboard = ResearchDashboard(history_length=500)
    collector = MetricsCollector(
        orchestrator_url=args.orch_url,
        learner_url=args.learner_url,
        policy_name=args.policy_name,
        baselines_path=args.baselines,
        rerun_enabled=args.rerun,
    )
    
    # Set up UI
    dashboard.setup_ui()
    
    if not dashboard.initialized:
        print("Dashboard initialization failed (Dear PyGui not available)")
        print("Install with: pip install dearpygui")
        return
    
    print("Dashboard running. Press Ctrl+C to exit.")
    
    # Main loop
    step = 0
    try:
        while dashboard.run_frame():
            step += 1
            metrics = collector.collect()
            
            # Update dashboard
            dashboard.update(metrics)

            # Add any live eureka events
            for ev in collector.pop_eureka_events():
                dashboard.add_eureka(ev)
            
            # Small delay to prevent CPU spinning
            time.sleep(0.016)  # ~60 FPS
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        dashboard.cleanup()
    
    print("Dashboard closed.")


if __name__ == "__main__":
    main()
