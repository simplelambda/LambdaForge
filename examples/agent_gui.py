"""Optional tiny desktop client of the same project-pinned facade, not an embedded agent.

Run: python agent_gui.py /absolute/consumer/project
Tk is optional and imported only by main. No LLM key, shell execution or cluster mutation.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from lambdaforge.agents import AgentServices


def main() -> None:
    import json
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText

    if len(sys.argv) != 2:
        raise ValueError("Provide the exact project root as the only argument.")
    services = AgentServices(sys.argv[1])
    pool = ThreadPoolExecutor(max_workers=1)
    window = tk.Tk()
    window.title("LambdaForge project evidence — read only")
    query = tk.Entry(window, width=70)
    query.pack(fill="x")
    result = ScrolledText(window, width=100, height=30)
    result.pack(fill="both", expand=True)

    def request() -> None:
        text = query.get()
        result.delete("1.0", "end")
        result.insert("end", "Searching pinned project knowledge…")
        future = pool.submit(services.knowledge_search, text)

        def collect() -> None:
            if not future.done():
                window.after(50, collect)
                return
            result.delete("1.0", "end")
            result.insert("end", json.dumps(asdict(future.result()), indent=2, ensure_ascii=False))

        window.after(50, collect)

    tk.Button(window, text="Search knowledge", command=request).pack()
    window.mainloop()
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
