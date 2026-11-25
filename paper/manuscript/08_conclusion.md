# Conclusion (manuscript section)

---

Holding the LLM, features, data and training budget fixed, we varied only the connector of an LLM-based vision-language
model. A learnable injection gate that starts at zero lowered worst-group accuracy by 10.6 and 25.2 points on
Qwen3.5-4B and by 9.7 and 25.5 points on Phi-4-mini, and made robustness depend on the seed. With Qwen3.5-9B the
watermark gap grew to 50.3 points. A gate that starts at
0.1 or higher behaved like a fixed gate, while one started at 0.01 fell back near zero and collapsed, so a start at
or near zero is the cause. The gate stays small because its large gradient
cancels itself, not because the gradient vanishes, and the visual pathway stays weak rather than closed. The effect
did not appear with per-layer gates or on open-ended VQA-CP answers, where strong injection hurt generation. Capping the
injection at the text norm gave the best free-form answers but gave back most of the robustness gain on the
watermark task at 4B (not at 9B), so injection strength can trade one kind of reliability against the other. Future work should test larger models trained for longer
and benchmarks of spurious cues for
open-ended answers [ye2026mmspubench; yang2025spuriverse].
