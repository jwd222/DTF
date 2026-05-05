import argparse
import sys

from drone_traffic.core.config import load_config
from drone_traffic.pipeline import PipelineManager


def main():
    parser = argparse.ArgumentParser(
        prog="drone-traffic",
        description="Multi-camera traffic monitoring system",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config.yaml"
    )

    subparsers.add_parser("check", help="Validate config and exit")

    api_parser = subparsers.add_parser("api", help="Run API server only")
    api_parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config.yaml"
    )
    api_parser.add_argument("--host", default=None)
    api_parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config if hasattr(args, "config") else "config.yaml")

    if args.command == "check":
        print("Config validated successfully.")
        sys.exit(0)

    if args.command == "run":
        manager = PipelineManager(config)
        manager.run()

    if args.command == "api":
        import uvicorn

        host = args.host or config.api.host
        port = args.port or config.api.port
        uvicorn.run(
            "drone_traffic.api.main:create_app",
            host=host,
            port=port,
            factory=True,
        )


if __name__ == "__main__":
    main()
