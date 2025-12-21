using System;
using System.Reflection;
using BepInEx.Logging;
using HarmonyLib;
using Il2CppInterop.Runtime;
using Unity.MLAgents;
using UnityEngine;

namespace Metabonk;

/// <summary>
/// Dynamic Harmony patches for IL2CPP games.
/// We avoid compile-time references to game types by locating them by name at runtime.
/// </summary>
public static class Hooks
{
    private static ManualLogSource? _log;

    public static void ApplyDynamicPatches(Harmony harmony, ManualLogSource log)
    {
        _log = log;

        // TODO: Update these type/method names after inspecting BepInEx/interop assemblies.
        PatchByName(harmony, "GameManager", "FixedUpdate", postfix: nameof(AfterFixedUpdate));
        PatchByName(harmony, "GameManager", "StartRun", postfix: nameof(AfterStartRun));
        PatchByName(harmony, "GameManager", "GameOver", postfix: nameof(AfterGameOver));
    }

    private static void PatchByName(
        Harmony harmony,
        string typeName,
        string methodName,
        string? prefix = null,
        string? postfix = null)
    {
        try
        {
            var il2cppType = Il2CppType.From(typeName);
            if (il2cppType == null)
            {
                _log?.LogWarning($"Type not found: {typeName}");
                return;
            }

            var type = il2cppType.ReflectionType;
            var method = AccessTools.Method(type, methodName);
            if (method == null)
            {
                _log?.LogWarning($"Method not found: {typeName}.{methodName}");
                return;
            }

            HarmonyMethod? pre = prefix != null
                ? new HarmonyMethod(typeof(Hooks).GetMethod(prefix, BindingFlags.Static | BindingFlags.NonPublic))
                : null;
            HarmonyMethod? post = postfix != null
                ? new HarmonyMethod(typeof(Hooks).GetMethod(postfix, BindingFlags.Static | BindingFlags.NonPublic))
                : null;

            harmony.Patch(method, prefix: pre, postfix: post);
            _log?.LogInfo($"Patched {typeName}.{methodName}");
        }
        catch (Exception e)
        {
            _log?.LogError($"Failed to patch {typeName}.{methodName}: {e}");
        }
    }

    private static void AfterFixedUpdate()
    {
        // Deterministic ML-Agents step.
        if (Academy.Instance != null)
            Academy.Instance.EnvironmentStep();
    }

    private static void AfterStartRun(object __instance)
    {
        try
        {
            // Locate player GameObject. Update field/property names per interop inspection.
            var playerField = AccessTools.Field(__instance.GetType(), "player");
            var playerObj = playerField?.GetValue(__instance) as GameObject;

            if (playerObj == null)
            {
                _log?.LogWarning("Player object not found on StartRun; update hook lookup.");
                return;
            }

            if (playerObj.GetComponent<BonkAgent>() == null)
            {
                playerObj.AddComponent<BonkAgent>();
                _log?.LogInfo("BonkAgent attached to player.");
            }
        }
        catch (Exception e)
        {
            _log?.LogError($"AfterStartRun failed: {e}");
        }
    }

    private static void AfterGameOver()
    {
        // Trigger episode end if needed.
        var agent = UnityEngine.Object.FindObjectOfType<BonkAgent>();
        if (agent != null)
            agent.EndEpisode();
    }
}

