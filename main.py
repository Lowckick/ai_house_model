"""
Smart House ML Control System — Main Entry Point

Usage:
  python main.py train          Train the agent (with live dashboard)
  python main.py train --no-ui  Train without UI (faster, logs to console)
  python main.py simulate       Run a live simulation using trained model
  python main.py report         Generate analytics report from training data
  python main.py screenshots    Generate dashboard screenshots for report
  python main.py demo           Quick 5-episode demo with dashboard
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_train(args):
    from smart_house.ml.trainer import Trainer
    trainer = Trainer(use_api_weather=not args.no_api)

    if args.no_ui:
        trainer.train(n_episodes=args.episodes, verbose=True)
    else:
        from smart_house.dashboard.display import LiveDashboard
        dash = LiveDashboard()
        dash.run_with_trainer(trainer, n_episodes=args.episodes)


def cmd_simulate(args):
    import time
    import datetime
    from rich.console import Console
    from rich.live import Live

    from smart_house.ml.trainer import Trainer
    from smart_house.ml.agent import encode_state, action_index_to_devices
    from smart_house.ml.rule_layer import RuleLayer
    from smart_house.environment.indoor_state import IndoorPhysicsSimulator, DeviceState
    from smart_house.environment.weather import WeatherProvider
    from smart_house.dashboard.display import LiveDashboard

    console = Console()
    console.print("[bold cyan]Starting Smart House Simulation (inference mode)...[/bold cyan]")

    trainer = Trainer(use_api_weather=not args.no_api)
    trainer.agent.epsilon = 0.0          # pure exploitation
    physics = IndoorPhysicsSimulator()
    weather_provider = WeatherProvider(use_api=not args.no_api)
    rule_layer = RuleLayer()
    dash = LiveDashboard()

    indoor = physics.reset()
    devices = DeviceState()
    weather = weather_provider.get()
    step = 0

    with Live(dash.build_layout(), refresh_per_second=2, screen=True) as live:
        while True:
            state = encode_state(indoor, weather, devices)
            action_idx = trainer.agent.select_action(state)
            raw_devices = action_index_to_devices(action_idx, devices)
            devices = rule_layer.apply(raw_devices, indoor, weather)

            if step % 30 == 0:
                weather = weather_provider.get()

            indoor = physics.step(weather, devices)
            reward = trainer.agent.reward_fn.compute(indoor, devices)

            dash.update(0, step, indoor, devices, weather, reward,
                        trainer.agent.epsilon)
            dash._source = weather_provider.source
            live.update(dash.build_layout())

            step += 1
            time.sleep(args.speed)


def cmd_report(args):
    from smart_house.analytics.reporter import generate_training_report, print_text_summary
    from smart_house.ml.trainer import DB_PATH
    import os

    db = os.path.abspath(DB_PATH)
    print_text_summary(db)
    path = generate_training_report(db)
    if path:
        print(f"Report image: {path}")


def cmd_screenshots(args):
    from smart_house.analytics.screenshots import generate_simulation_screenshots

    paths = generate_simulation_screenshots(
        output_dir=args.output,
        count=args.count,
        use_api_weather=not args.no_api,
        steps=args.steps,
    )
    if not paths:
        print("No screenshots were generated.")
        return
    print("Generated screenshots:")
    for path in paths:
        print(f"  {path}")


def cmd_demo(args):
    args.episodes = 5
    args.no_api = False
    cmd_train(args)


def main():
    parser = argparse.ArgumentParser(
        description="Smart House ML Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train the DQN agent")
    p_train.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p_train.add_argument("--no-ui", action="store_true", help="Disable live dashboard")
    p_train.add_argument("--no-api", action="store_true", help="Skip weather API, use synthetic data")

    # simulate
    p_sim = sub.add_parser("simulate", help="Live simulation (inference only)")
    p_sim.add_argument("--no-api", action="store_true", help="Skip weather API")
    p_sim.add_argument("--speed", type=float, default=1.0, help="Seconds between steps")

    # report
    p_rep = sub.add_parser("report", help="Generate training analytics report")

    # screenshots
    p_shot = sub.add_parser("screenshots", help="Generate dashboard screenshots for the coursework report")
    p_shot.add_argument("--count", type=int, default=4, help="Number of PNG screenshots to generate")
    p_shot.add_argument("--steps", type=int, default=144, help="Number of simulated steps")
    p_shot.add_argument("--no-api", action="store_true", help="Skip weather API")
    p_shot.add_argument("--output", default=None, help="Output directory")

    # demo
    p_demo = sub.add_parser("demo", help="Quick 5-episode demo with dashboard")
    p_demo.add_argument("--no-api", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "screenshots":
        cmd_screenshots(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
