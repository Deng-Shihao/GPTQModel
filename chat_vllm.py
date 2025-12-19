import argparse
import time

from vllm import LLM, SamplingParams


def _fmt_sec(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM interactive chat + speed stats")
    parser.add_argument(
        "-m",
        "--model",
        help="Model name or local path",
    )
    parser.add_argument(
        "--tp",
        "--tensor-parallel-size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps (0 to disable)",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=128,
        help="Warmup max tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Sampling top_p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per assistant response",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Set trust_remote_code=True when loading the model",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args() 

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=False,
    )
    print(
        "[config] "
        f"model={args.model!r} "
        f"tp={args.tensor_parallel_size} "
        f"temperature={args.temperature} "
        f"top_p={args.top_p} "
        f"max_tokens={args.max_tokens} "
        f"warmup_steps={args.warmup_steps}",
        flush=True,
    )

    # Warm up (reduces first-turn latency)
    if args.warmup_steps > 0:
        warmup_messages: list[dict[str, str]] = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": "Hello"},
        ]
        warmup_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_max_tokens,
        )
        print(f"[warmup] starting ({args.warmup_steps} steps)...", flush=True)
        for i in range(args.warmup_steps):
            print(f"[warmup] {i + 1}/{args.warmup_steps} running...", flush=True)
            step_t0 = time.perf_counter()
            warmup_outputs = llm.chat(
                warmup_messages,
                sampling_params=warmup_sampling_params,
                use_tqdm=False,
            )
            step_elapsed_s = time.perf_counter() - step_t0
            step_gen_tokens = len(warmup_outputs[0].outputs[0].token_ids or [])
            step_tok_s = (
                step_gen_tokens / step_elapsed_s if step_elapsed_s > 0 else 0.0
            )
            print(
                f"[warmup] {i + 1}/{args.warmup_steps} "
                f"elapsed={_fmt_sec(step_elapsed_s)} "
                f"gen={step_gen_tokens} tok/s={step_tok_s:.2f}",
                flush=True,
            )
        print("[warmup] done.", flush=True)

    # Interactive chat (OpenAI-style messages)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": args.system},
    ]
    print("Chat started. Commands: /exit, /quit, /reset")
    total_gen_tokens = 0
    total_time_s = 0.0
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/reset":
            messages = messages[:1]
            print("(history cleared)")
            continue

        messages.append({"role": "user", "content": user_text})
        t0 = time.perf_counter()
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        elapsed_s = time.perf_counter() - t0
        assistant_text = outputs[0].outputs[0].text.strip()
        print(f"Assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})

        out0 = outputs[0]
        prompt_tokens = len(out0.prompt_token_ids or [])
        gen_tokens = len(out0.outputs[0].token_ids or [])
        cached_tokens = out0.num_cached_tokens or 0

        metrics = out0.metrics
        ttft_s: float | None = None
        queue_s: float | None = None
        prefill_s: float | None = None
        decode_s: float | None = None
        if metrics is not None:
            if hasattr(metrics, "first_token_latency"):
                ttft_s = getattr(metrics, "first_token_latency", None)
            elif hasattr(metrics, "first_token_time") and hasattr(
                metrics, "arrival_time"
            ):
                first_token_time = getattr(metrics, "first_token_time", None)
                arrival_time = getattr(metrics, "arrival_time", None)
                if first_token_time is not None and arrival_time is not None:
                    ttft_s = first_token_time - arrival_time

            if hasattr(metrics, "time_in_queue"):
                queue_s = getattr(metrics, "time_in_queue", None)
            elif hasattr(metrics, "queued_ts") and hasattr(metrics, "scheduled_ts"):
                queued_ts = getattr(metrics, "queued_ts", 0.0) or 0.0
                scheduled_ts = getattr(metrics, "scheduled_ts", 0.0) or 0.0
                if queued_ts and scheduled_ts:
                    queue_s = scheduled_ts - queued_ts

            if hasattr(metrics, "scheduled_ts") and hasattr(metrics, "first_token_ts"):
                scheduled_ts = getattr(metrics, "scheduled_ts", 0.0) or 0.0
                first_token_ts = getattr(metrics, "first_token_ts", 0.0) or 0.0
                if scheduled_ts and first_token_ts:
                    prefill_s = first_token_ts - scheduled_ts

            if hasattr(metrics, "first_token_ts") and hasattr(metrics, "last_token_ts"):
                first_token_ts = getattr(metrics, "first_token_ts", 0.0) or 0.0
                last_token_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0
                if first_token_ts and last_token_ts:
                    decode_s = last_token_ts - first_token_ts

        total_gen_tokens += gen_tokens
        total_time_s += elapsed_s

        gen_tok_s = (gen_tokens / elapsed_s) if elapsed_s > 0 else 0.0
        avg_gen_tok_s = (total_gen_tokens / total_time_s) if total_time_s > 0 else 0.0
        print(
            "[speed] "
            f"prompt={prompt_tokens} (cached={cached_tokens}) "
            f"gen={gen_tokens} "
            f"elapsed={_fmt_sec(elapsed_s)} "
            f"ttft={_fmt_sec(ttft_s)} "
            f"queue={_fmt_sec(queue_s)} "
            f"prefill={_fmt_sec(prefill_s)} "
            f"decode={_fmt_sec(decode_s)} "
            f"gen_tok/s={gen_tok_s:.2f} "
            f"avg_gen_tok/s={avg_gen_tok_s:.2f}"
        )


if __name__ == "__main__":
    # vLLM internally uses multiprocessing "spawn"; keep the main guard.
    import multiprocessing as mp

    mp.freeze_support()
    main()
