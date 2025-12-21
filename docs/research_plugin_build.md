# ResearchPlugin build + install

This plugin provides deterministic stepping + 84×84 RGB24 shared memory IPC.

## Prereqs
- MegaBonk installed with **BepInEx 6 IL2CPP** already set up.
- .NET SDK 8+ installed (`dotnet --version`).

## Build + install (Linux / macOS / WSL)

From the MetaBonk repo:

```
chmod +x scripts/build_research_plugin.sh
scripts/build_research_plugin.sh "/path/to/MegaBonk"
```

This:
1. Builds `mods/ResearchPlugin.csproj` with `IL2CPP` define.
2. Outputs `dist/research_plugin/ResearchPlugin.dll`.
3. Copies it to `MegaBonk/BepInEx/plugins/ResearchPlugin.dll`.

## Notes
- The build references Unity/BepInEx assemblies from your game folder via the MSBuild property
  `MEGABONK_DIR`. If your Managed folder is in a different place, edit
  `mods/ResearchPlugin.csproj` HintPaths.
- The Mono version `mods/ResearchPlugin.cs` is excluded automatically when `IL2CPP` is defined.

