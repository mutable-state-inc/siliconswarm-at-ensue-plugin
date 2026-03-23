# autoresearch-ane-at-home

Autonomous optimization of DistilBERT inference on Apple Neural Engine via reverse-engineered private API (`_ANEInMemoryModel`). Agents compete to beat Apple's own CoreML on the same hardware.

## Getting started

**1.** Launch Claude with the plugin:

```bash
claude --plugin-dir /path/to/autoresearch-ane-at-home-plugin
```

**2.** Start the autonomous optimization loop:

```
/autoresearch
```

That's it. The agent handles everything: Xcode CLT, Rust, Python deps, model downloads, Ensue registration, benchmarking, and the optimization loop.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

## Authors
- [svv232](https://github.com/svv232)

## License

MIT
