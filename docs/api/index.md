# API Reference

This section provides detailed API documentation for all public modules in metalab.

## Core

- [Experiment](experiment.md) - Main experiment orchestration class
- [Operations](operation.md) - Define computational units with the `@operation` decorator
- [Context Specs](context.md) - Declare data dependencies and file paths
- [Parameters](params.md) - Parameter space definitions for grid, random, and manual search
- [Seeds](seeds.md) - Reproducible random number generation

## Runtime

- [Capture](capture.md) - Emit metrics, artifacts, and logs during execution
- [Results](results.md) - Query and analyze experiment results

## Infrastructure

- [Storage](stores.md) - Persist runs and artifacts to file or database backends
- [Executors](executors.md) - Run experiments locally or on distributed systems
