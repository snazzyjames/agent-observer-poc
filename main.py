import argparse
from pipeline import build_pipeline, PipelineState
from observer import Observer

DEFAULT_TASK = (
    "Build a REST API endpoint that validates user input, "
    "stores data in a database, and returns a JSON response."
)

SEP = "=" * 60


def display_step(step: int, node: str, report: dict) -> None:
    print(f"\n{SEP}")
    print(f"Step {step:2d}  |  Node: {node}")
    print(SEP)

    drift = report.get("drift_score")
    if drift is not None:
        flag = "  <-- degrading" if drift < 0.7 else ""
        print(f"Drift score:  {drift:.3f}{flag}")

    tokens = report.get("context_tokens")
    if tokens is not None:
        print(f"Context:      {tokens:,} tokens")

    prompt = report.get("system_prompt", "")
    if prompt:
        preview = prompt[:120].replace("\n", " ")
        suffix = "..." if len(prompt) > 120 else ""
        print(f"System:       {preview}{suffix}")

    reasoning = report.get("reasoning", "")
    if reasoning:
        print(f"\nReasoning:\n{reasoning}")


def print_summary(step: int, observer: Observer) -> None:
    print(f"\n{SEP}")
    print("Summary")
    print(SEP)
    print(f"Steps run:    {step}")

    scores = observer.drift_scores()
    if scores:
        print(f"Final drift:  {scores[-1]:.3f}")
        if len(scores) > 1:
            delta = scores[-1] - scores[0]
            direction = "degraded" if delta < -0.05 else "stable"
            print(f"Drift trend:  {delta:+.3f} ({direction})")
    else:
        print("Drift:        no data")

    peak = observer.peak_context_tokens()
    if peak:
        print(f"Peak context: {peak:,} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Observer: observability for agentic pipelines"
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task specification")
    parser.add_argument("--iterations", type=int, default=4, help="Pipeline iterations")
    args = parser.parse_args()

    print("Agent Observer POC — Agentic Pipeline Observability")
    print(f"Task:       {args.task}")
    print(f"Iterations: {args.iterations}")

    observer = Observer(task_spec=args.task)
    graph = build_pipeline(
        task=args.task,
        max_iterations=args.iterations,
        observer=observer,
    )

    initial_state: PipelineState = {
        "task": args.task,
        "messages": [{"role": "user", "content": args.task}],
        "iteration": 0,
        "system_prompt": "",
        "reasoning": "",
    }

    step = 0
    for event in graph.stream(initial_state):
        for node_name, state in event.items():
            if state.get("reasoning") == "":
                continue
            step += 1
            report = observer.get_report(node_name, state)
            display_step(step, node_name, report)

    print_summary(step, observer)


if __name__ == "__main__":
    main()
