"""
HR Analytics — CLI Entry Point.

Provides convenient commands for running the system locally.
"""
import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

def main() -> None:
    """Route CLI commands to appropriate subsystems."""
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|serve]")
        print("  train  — Run the full ML training pipeline")
        print("  serve  — Start the FastAPI prediction server")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        from src.train_attrition_model import main as train_main
        train_main()
    elif command == "serve":
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
