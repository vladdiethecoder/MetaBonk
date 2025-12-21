using BepInEx;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using Unity.MLAgents;
using UnityEngine;

namespace Metabonk;

[BepInPlugin("ai.metabonk.plugin", "MetabonkPlugin", "0.1.0")]
public sealed class MetabonkPlugin : BasePlugin
{
    public static MetabonkPlugin? Instance { get; private set; }
    private Harmony? _harmony;

    public override void Load()
    {
        Instance = this;
        Log.LogInfo("MetabonkPlugin loading (IL2CPP)");

        // Disable automatic stepping so we can step in FixedUpdate for determinism.
        if (Academy.Instance != null)
            Academy.Instance.AutomaticSteppingEnabled = false;

        _harmony = new Harmony("ai.metabonk.plugin.harmony");
        Hooks.ApplyDynamicPatches(_harmony, Log);
        Log.LogInfo("MetabonkPlugin loaded");
    }
}

