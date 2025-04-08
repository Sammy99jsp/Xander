# 🚧 Xander: A D&D Combat Environment for Reinforcement Learning

Hello, this is a university project that attempts to make an RL environment for D&D 5E combat,
where agents can play the role of monsters against players.

This is still very much in progress.

## Features

Xander supports:
- [x] Dice Notation:
  - [x] `XdY` epxressions
  - [x] Modifiers
  - [x] Advantage/Disadvantage (`2d20kl1`, `2d20kh1`) 
- [x] Stat Blocks:
  - [x] Loading from JSON
  - [x] Ability Scores, Modifiers, and Skills (with proficiencies)
  - [x] HP, Temporary HP
  - [x] Checks, Saving Throws
  - [ ] Death Saves (in progress)
- [ x] Visualization:
  - [x] Combat arena map
  - [x] `_repr_html_` HTML display for Jupyter
- [ ] Reference Agents:
  - [ ] Q-Learning (in progress)

## Building


### Prerequisites
I still have to get this whole CI/CD thing figured out for Linux, but builds are available for MacOS and Windows.

To build from scratch, ensure you have:
* Rust toolchain: `rustup`, `cargo` with a nightly compiler (`rustup install nightly`)
* A new `conda` environment set up with Python 3.12 (`conda create -n xander python=3.12`)
* Maturin (`cargo install maturin`)
* Wasm-pack (`cargo install wasm-pack`)

### Building the Project

* Run `maturin build` to build a Python `.whl` file.
* Run `wasm-pack build -- --features web` to build for wasm.

## Implementation
* The core environment is written in Rust, and exposed to python via [pyo3](https://github.com/PyO3/pyo3).
  * Python type bindings are available, but they do have to be manually updated (WIP).
* TODO: write more here.

### Examples
```python
from xander.engine import dice
from xander.engine.actors import Stats

dice.random_seed()

rat = State.from_json("tests/rat.json")
print(rat)
```

### Explore Why this initial DQN didn't work.
* High variance in Q-values -> Dueling
* Spare Rewards -> Prioritized Replay Buffer.

### Notes
Put the reward functions for your environments.

5 runs for the confidence interval.