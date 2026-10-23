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
    import datetime
    import torch
    from pathlib import Path
    from smart_house.ml.experiments import run_experiment
    torch.set_num_threads(args.threads)
    seeds = args.seeds or [42]
    output = args.output_dir or str(Path(__file__).resolve().parent / 'data/user/training' /
             datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    callback = None
    if not args.no_ui and len(seeds) == 1:
        from smart_house.dashboard.display import LiveDashboard
        callback = lambda trainer, episodes: LiveDashboard().run_with_trainer(trainer, episodes)
    run_experiment(output, seeds=seeds, episodes=args.episodes, steps=args.steps,
                   warmup_episodes=args.warmup_episodes, warmup_epochs=args.warmup_epochs,
                   validation_interval=args.validation_interval, validation_episodes=args.validation_episodes,
                   test_episodes=args.test_episodes, resume=args.resume,
                   reference_model=args.reference_model, train_callback=callback)


def cmd_simulate(args):
    import time
    from rich.console import Console
    from rich.live import Live

    from smart_house.ml.agent import encode_state
    from smart_house.ml.rule_layer import RuleLayer
    from smart_house.environment.indoor_state import IndoorPhysicsSimulator, DeviceState
    from smart_house.environment.weather import WeatherProvider
    from smart_house.dashboard.display import LiveDashboard

    console = Console()
    console.print("[bold cyan]Starting Smart House Simulation (inference mode)...[/bold cyan]")

    from smart_house.analytics.comparison import load_agent
    agent = load_agent(args.model)
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
            action_idx = agent.select_action(state)
            raw_devices = agent.decode_action(action_idx, devices, indoor, weather)
            devices = rule_layer.apply(raw_devices, indoor, weather)

            if step % 30 == 0:
                weather = weather_provider.get()

            indoor = physics.step(weather, devices)
            reward = agent.reward_fn.compute(indoor, devices)

            dash.update(0, step, indoor, devices, weather, reward,
                        agent.epsilon)
            dash._source = weather_provider.source
            live.update(dash.build_layout())

            step += 1
            time.sleep(args.speed)


def cmd_report(args):
    from smart_house.analytics.reporter import generate_training_report, print_text_summary
    from smart_house.ml.trainer import DB_PATH
    import os

    db = os.path.abspath(args.db or DB_PATH)
    print_text_summary(db)
    path = generate_training_report(db, args.output) if args.output else generate_training_report(db)
    if path:
        print(f"Report image: {path}")


def cmd_screenshots(args):
    from smart_house.analytics.screenshots import generate_simulation_screenshots

    paths = generate_simulation_screenshots(
        output_dir=args.output,
        count=args.count,
        use_api_weather=not args.no_api,
        steps=args.steps,
        model_path=args.model,
    )
    if not paths:
        print("No screenshots were generated.")
        return
    print("Generated screenshots:")
    for path in paths:
        print(f"  {path}")


def cmd_import_data(args):
    import json
    from pathlib import Path
    from smart_house.analytics.history import History
    from smart_house.analytics.ingestion import ImportService, ValidationError
    from smart_house.environment.readings import MAX_FILE_BYTES

    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        raise ValueError("Input and output paths must be different")
    if target.exists():
        raise ValueError("Output already exists; choose a new --output path")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds 5 MiB")
    try:
        payload = ImportService(args.model, History(args.db)).import_text(
            source.name, source.read_text(encoding="utf-8-sig"))
    except ValidationError as exc:
        raise ValueError(json.dumps(exc.report, ensure_ascii=False)) from exc
    write_json(target, payload)
    print(f"Imported {payload['count']} readings (history #{payload['id']}). Recommendations saved to: {target}")


def write_json(path, payload):
    import json
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.write("\n")


def cmd_validate_data(args):
    import json
    from pathlib import Path
    from smart_house.environment.readings import inspect_readings, MAX_FILE_BYTES
    source = Path(args.input)
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds 5 MiB")
    _, report = inspect_readings(source.read_text(encoding="utf-8-sig"), source.suffix)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['valid']:
        raise ValueError("File validation failed; see report above")


def cmd_evaluate(args):
    from pathlib import Path
    from smart_house.analytics.ingestion import ImportService
    from smart_house.analytics.history import History
    if Path(args.output).exists():
        raise ValueError("Output already exists; choose a new --output path")
    payload = ImportService(args.model, History(args.db)).evaluate(args.episodes, args.steps, args.seed)
    write_json(args.output, payload)
    print(f"Evaluation #{payload['id']} saved to: {args.output}")


def cmd_demo(args):
    # Reuse all training defaults for the demo command.
    main(["train", "--episodes", "5"] + (["--no-ui"] if args.no_ui else []))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Smart House ML Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")
    default_model = os.path.join(os.path.dirname(__file__), "data", "model.pt")

    # train
    p_train = sub.add_parser("train", help="Train the DQN agent")
    p_train.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p_train.add_argument("--no-ui", action="store_true", help="Disable live dashboard")
    p_train.add_argument("--no-api", action="store_true", help="Compatibility flag: training always uses seasonal synthetic weather")
    p_train.add_argument("--seeds", type=int, nargs="+", help="Distinct training seeds; default: 42")
    p_train.add_argument("--steps", type=int, default=144)
    p_train.add_argument("--output-dir", help="New experiment directory; original data/model.pt is preserved")
    p_train.add_argument("--warmup-episodes", type=int, default=12)
    p_train.add_argument("--warmup-epochs", type=int, default=5)
    p_train.add_argument("--validation-interval", type=int, default=5)
    p_train.add_argument("--validation-episodes", type=int, default=6)
    p_train.add_argument("--test-episodes", type=int, default=12)
    p_train.add_argument("--threads", type=int, choices=range(1, 17), default=2)
    p_train.add_argument("--resume", help="Warm start from a residual checkpoint; replay buffer is regenerated")
    p_train.add_argument("--reference-model", help="Optional old checkpoint to evaluate on identical test scenarios")

    # simulate
    p_sim = sub.add_parser("simulate", help="Live simulation (inference only)")
    p_sim.add_argument("--no-api", action="store_true", help="Skip weather API")
    p_sim.add_argument("--speed", type=float, default=1.0, help="Seconds between steps")

    p_sim.add_argument("--model", default=default_model)

    # report
    p_rep = sub.add_parser("report", help="Generate training analytics report")
    p_rep.add_argument("--db", help="Training SQLite database; defaults to data/training.db")
    p_rep.add_argument("--output", help="Directory for generated reports")

    # screenshots
    p_shot = sub.add_parser("screenshots", help="Generate dashboard screenshots for the coursework report")
    p_shot.add_argument("--count", type=int, default=4, help="Number of PNG screenshots to generate")
    p_shot.add_argument("--steps", type=int, default=144, help="Number of simulated steps")
    p_shot.add_argument("--no-api", action="store_true", help="Skip weather API")
    p_shot.add_argument("--output", default=None, help="Output directory")
    p_shot.add_argument("--model", default=default_model)

    default_db = os.path.join(os.path.dirname(__file__), "data", "user", "measurements.db")
    p_import = sub.add_parser("import-data", help="Validate, store indoor readings and generate recommendations")
    p_import.add_argument("--input", required=True, help="UTF-8 JSON, CSV or TXT file")
    p_import.add_argument("--output", required=True, help="New JSON output file (never overwritten)")
    p_import.add_argument("--model", default=default_model, help="Trusted trained PyTorch checkpoint")
    p_import.add_argument("--db", default=default_db, help="Separate measurement history SQLite database")

    p_validate = sub.add_parser("validate-data", help="Report all file errors and warnings without importing")
    p_validate.add_argument("--input", required=True)

    p_eval = sub.add_parser("evaluate", help="Compare DQN with deterministic control in paired simulation")
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--steps", type=int, default=144)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--model", default=default_model)
    p_eval.add_argument("--db", default=default_db)
    p_eval.add_argument("--output", required=True)

    p_web = sub.add_parser("web", help="Start browser upload, history and evaluation interface")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--model", default=default_model)
    p_web.add_argument("--db", default=default_db)

    # demo
    p_demo = sub.add_parser("demo", help="Quick 5-episode demo with dashboard")
    p_demo.add_argument("--no-api", action="store_true")
    p_demo.add_argument("--no-ui", action="store_true")

    args = parser.parse_args(argv)

    if args.command in {"import-data", "validate-data", "evaluate", "web"}:
        try:
            if args.command == "web":
                from smart_house.web.server import serve
                serve(args.host, args.port, args.model, args.db)
            else:
                {"import-data": cmd_import_data, "validate-data": cmd_validate_data,
                 "evaluate": cmd_evaluate}[args.command](args)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            parser.exit(2, f"Command failed: {exc}\n")
    elif args.command == "train":
        try:
            cmd_train(args)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"Training failed: {exc}\n")
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
