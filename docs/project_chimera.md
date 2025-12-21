# Project Chimera: A Generalist Neuro-Genie Stack for Universal Game Mastery

## 1. Executive Summary: The "Universal Ludology" Engine

The evolution of MetaBonk from a specialized speedrunner into a generalist AI stack requires a fundamental architectural pivot. Instead of hard-coding the agent to "press W to move," we must build an agent that understands the concept of movement and intuitively maps it to whatever controls the current game requires.

This proposed architecture, **Project Chimera**, leverages Genie 3's generative world modeling[^1] and AdaWorld's latent action adaptation[^3] to achieve a **"Watch, Dream, and Conquer"** pipeline.

**The core evolution:**

- **From code-access to visual-only:** The agent will no longer read memory addresses. It will learn purely from pixels, treating the game screen as its retina.
- **From hard-coded keys to "latent intuition":** The agent will not output `Key_W`. It will output a high-level latent action (e.g., `Intent_Navigate_Forward`). A lightweight **Reflex Layer** will translate this intuition into the specific keys for the game at hand (WASD for Minecraft, mouse clicks for Balatro).
- **Target domain expansion:** The stack is designed to master MegaBonk (the hurdle), then scale to Minecraft (3D voxel), Factorio/Satisfactory (logistics/optimization), YOMIH (high-speed fighting), and Balatro (discrete/stochastic strategy).

## 2. The Core Challenge: "Intuition" Without Dumbing Down

To give the project the ability to "intuit" actions without simplifying them, separate **strategic intent (System 2)** from **motor execution (System 1)**.

### 2.1 The Solution: Latent Action Models (LAMs)

We utilize research from Genie 3[^4] and AdaWorld[^3] to create a latent action space.

- **The concept:** When humans play, we don't think "contract finger muscle 30%." We think "Jump." The muscle movement is automatic.
- **The implementation:** Train a latent action model (LAM) on thousands of hours of video (e.g., YouTube gameplay of the target games). The LAM analyzes the video and learns that "when the screen moves like this, an action occurred." It assigns a mathematical vector (a latent token) to that action.
- **The result:** The agent learns a library of **visual intentions** (e.g., latent action 42 = Jump, latent action 105 = Open Inventory). This is the "intuition."

## 3. The New Architecture: Three-Phase Evolution

To turn MetaBonk into a generalist stack, implement a three-phase learning pipeline: **Watch, Dream, and Ground**.

### Phase 1: Watch & Intuit (The Cold Start)

**Input:** Massive unstructured video datasets (e.g., YouTube) of MegaBonk, Minecraft, Balatro, etc.

**Technology:** Genie 3 video tokenizer + VQ-VAE.

**Process:**

1. The model watches the video and compresses frames into discrete tokens.
2. It creates a latent action library by asking: "What hidden variable caused Frame A to become Frame B?"[^4]

**Result:** The agent acquires a "vocabulary of motion" for each game without ever pressing a key. It knows what is possible (jumping, shooting, dragging cards), but not how to trigger it yet.

### Phase 2: The Dream Dojo (Generative Training)

**Input:** The latent action library from Phase 1.

**Technology:** Genie 3 inference (FP4 on Blackwell) + DIAMOND diffusion.[^6]

**Process:**

- The agent enters the Dream Bridge (the neural simulator). It is not playing the real game; it is playing inside Genie 3's hallucination.
- **Promptable curriculum:** System 2 prompts the dream: "Generate a Balatro scenario with a high-stakes boss blind" or "Generate a Minecraft parkour map with ice."[^1]
- **Training:** The agent uses reinforcement learning to select the correct latent actions to maximize reward in the dream.

**Result:** A "master strategist" policy that knows what to do (e.g., "I need to jump now") but communicates in abstract latent thoughts.

### Phase 3: Grounding (The "Rosetta Stone" Adapter)

**Input:** The master strategist policy + the real game.

**Technology:** Action adapter (inverse dynamics)[^3] + liquid reflex layer.

This is where the key-binding problem is solved.

- **Few-shot calibration:** Let the agent observe a human playing for ~5 minutes with key-logging enabled.
- **The mapping:** The action adapter learns a translation such as:
  - Minecraft context: latent action 42 (Forward) → `Key_W`
  - Balatro context: latent action 42 (Select) → `Mouse_Click(x, y)`
  - YOMIH context: latent action 99 (Block) → key chord / sequence

**Why this works:** The complex "brain" (System 2) stays general. It always outputs "latent action 42". The lightweight "spinal cord" (System 1 / liquid network) handles the specific translation for the game. This allows switching games by swapping the adapter rather than retraining the whole brain.

## 4. Technical Stack Implementation

### 4.1 The Brain (System 2): The "Ludology" VLM

**Model:** Fine-tune a vision-language-action (VLA) model (SIMA-like or a LLaMA-V variant) on game strategy guides and wikis.

**Role:** High-level reasoning. It looks at the screen and decides: "I need to build a factory" (Factorio) or "I need to play a flush" (Balatro).

**Output:** Semantic subgoals and prompts for System 1.

### 4.2 The Body (System 1): Liquid Neural Networks (LNN)

**Role:** The pilot. Use Closed-form Continuous-time (CfC) networks.

**Why LNN:** Continuous-time modeling smooths jittery latent actions into fluid, human-like mouse movements.

**Function:** Receive latent action intent and emit motor commands (e.g., hold `W` for 2.5s, then tap `Space`; move mouse smoothly; click at `x, y`).

### 4.3 The "Universal" Input API (Matrix-Game 2.0 Integration)

To handle games with vastly different inputs (mouse vs keyboard vs controller), integrate an action-injection style interface (Matrix-Game 2.0).[^10]

**Modality experts:** Train three distinct reflex decoders:

- **FPS decoder:** Maps latent intents → WASD + mouse delta (Minecraft, MegaBonk).
- **Cursor decoder:** Maps latent intents → screen coordinates + click (Balatro, Factorio menus).
- **Combo decoder:** Maps latent intents → precise frame-perfect key sequences (YOMIH).

**Dynamic switching:** The agent detects the game type (via VLM analysis) and hot-swaps the correct reflex decoder.

## 5. Roadmap: From MegaBonk to Balatro

### Step 1: The "MegaBonk" Hurdle (Proof of Concept)

- Train Genie 3-style world model on MegaBonk video.
- Learn "movement intuition" (momentum, gravity) in the dream.
- Deploy with FPS decoder.
- **Goal:** Beat the game using only visual inputs.

### Step 2: The "Minecraft" Expansion (3D Generalization)

- Train on Minecraft YouTube dataset.
- Transfer the FPS decoder from MegaBonk.
- **Hypothesis:** Zero-shot navigation improvements, because "moving forward" shares latent structure across first-person games.

### Step 3: The "Balatro" Pivot (Abstract Logic)

This is the hardest shift (3D → 2D).

- Train on Balatro video.
- Swap to cursor decoder.
- System 2 must shift from spatial reasoning to probabilistic reasoning.
- **Crucial tech:** Promptable world events. Use Genie 3 to generate "impossible" Balatro hands to train probability-calculation failure cases.

### Step 4: The "Factorio" Grand Challenge (Long-Horizon)

- Integrate long-context memory (Ring Attention; 1M+ tokens).
- Correlate "place a belt" (action) with "resource arrives 10 minutes later" (delayed reward).

## 6. Summary of Capabilities

| Feature | Old Apex Protocol | New Chimera Protocol |
|---|---|---|
| Input | Game State (RAM/API) | Pure Vision (Pixels) |
| Action | Hard-coded (Press `W`) | Latent Intuition (Intent "Move") |
| Training | Real-time Game Engine | Accelerated Dream (Genie 3-style) |
| Adaptability | Single game hardcoded | Universal (swappable adapters) |
| Control | Brittle scripts | Liquid reflexes (human-like) |

By building this latent action interface, the agent effectively gains a **universal controller**. It learns to play the video of the game first (intuitive causality), then plugs that intuition into the keyboard/mouse when it’s time to act.

## Appendix A: Repo Mapping (MetaBonk)

Conceptually, Project Chimera aligns with existing Neuro-Genie components already present in this repo:

- Latent action modeling: `src/neuro_genie/latent_action_model.py`
- Latent→explicit translation: `src/neuro_genie/action_adapter.py`
- Reflex decoders / modality adapters: `src/neuro_genie/reflex_decoders.py`
- System 2 curriculum prompting: `src/neuro_genie/dungeon_master.py`
- World model (Genie-style): `src/neuro_genie/generative_world_model.py`
- Dream bridge wrapper: `src/neuro_genie/dream_bridge.py`
- Liquid stabilization / pilot: `src/neuro_genie/liquid_stabilizer.py`

---

[^1]: TODO: Add citation source (Genie 3 / generative interactive environments).
[^3]: TODO: Add citation source (AdaWorld / latent action adaptation).
[^4]: TODO: Add citation source (Genie-style latent actions from unlabeled video).
[^6]: TODO: Add citation source (DIAMOND diffusion world models).
[^10]: TODO: Add citation source (Matrix-Game 2.0 action injection).
