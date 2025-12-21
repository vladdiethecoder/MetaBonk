# MetabonkPlugin (BepInEx IL2CPP)

Recovery scaffold for the Unity-side interface layer.

## Build
1. Install BepInEx 6 IL2CPP for MegaBonk and run the game once to generate `BepInEx/interop/*` proxy DLLs.
2. Add references in your local build setup to:
   - `BepInEx.Core.dll`
   - `BepInEx.Unity.IL2CPP.dll`
   - `HarmonyLib.dll`
   - `Unity.MLAgents.dll`
   - `UnityEngine*.dll`
   - `BepInEx/interop/Assembly-CSharp.dll`
3. Build the project:
   ```bash
   dotnet build -c Release
   ```
4. Copy `bin/Release/net6.0/MetabonkPlugin.dll` to `BepInEx/plugins/`.

## Next Steps
- Inspect the generated interop assemblies to update `Hooks` type/method names and `BonkAgent` reflection helpers.
- Replace input translation in `BonkAgent.ApplyMovement/ApplyJump/...` with direct writes to the game's input structs.
- Expand observation stack to match the roadmap table precisely.

