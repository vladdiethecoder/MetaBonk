#if IL2CPP

// BepInEx Research Mod for MegaBonk (IL2CPP build).
// Mirrors the shared-memory + deterministic stepping protocol used by
// `src/bridge/research_shm.py`.

using BepInEx;
using BepInEx.Unity.IL2CPP;
using BepInEx.Logging;
using HarmonyLib;
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Reflection;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace MegabonkResearch
{
    [BepInPlugin(PluginInfo.PLUGIN_GUID, PluginInfo.PLUGIN_NAME, PluginInfo.PLUGIN_VERSION)]
    public class ResearchPluginIL2CPP : BasePlugin
    {
        internal static ManualLogSource Log;
        internal static Harmony harmony;
        internal static ResearchUpdater Updater;

        // Shared memory
        internal static MemoryMappedFile shm;
        internal static MemoryMappedViewAccessor accessor;
        internal static string shmName;

        // Flags/layout
        internal const int OBS_WIDTH = 84;
        internal const int OBS_HEIGHT = 84;
        internal const int HEADER_SIZE = 0x20;
        internal const int OBS_SIZE = OBS_WIDTH * OBS_HEIGHT * 3;
        internal const int ACTION_SIZE = 6 * 4;
        internal const int TOTAL_SIZE = HEADER_SIZE + OBS_SIZE + ACTION_SIZE;

        internal const int FLAG_WAIT = 0;
        internal const int FLAG_ACTION_READY = 1;
        internal const int FLAG_OBS_READY = 2;
        internal const int FLAG_RESET = 3;
        internal const int FLAG_TERMINATED = 4;

        internal static bool deterministicMode = false;

        public override void Load()
        {
            Log = base.Log;
            Log.LogInfo("=== MegaBonk Research Infrastructure (IL2CPP) ===");

            string instanceId = Environment.GetEnvironmentVariable("MEGABONK_INSTANCE_ID") ?? "0";
            shmName = $"megabonk_env_{instanceId}";
            deterministicMode = Environment.GetEnvironmentVariable("MEGABONK_DETERMINISTIC") == "1";

            InitSharedMemory();
            UnlockAllContent();

            harmony = new Harmony(PluginInfo.PLUGIN_GUID);
            harmony.PatchAll();

            // Attach updater MonoBehaviour to drive per-frame capture.
            Updater = AddComponent<ResearchUpdater>();
            Updater.enabled = true;

            Log.LogInfo($"Research IL2CPP plugin initialized (instance: {instanceId})");
        }

        public override bool Unload()
        {
            try
            {
                accessor?.Dispose();
                shm?.Dispose();
            }
            catch { }
            try
            {
                harmony?.UnpatchSelf();
            }
            catch { }
            return true;
        }

        internal static void InitSharedMemory()
        {
            try
            {
                shm = MemoryMappedFile.OpenExisting(shmName);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Connected to shared memory: {shmName}");
            }
            catch (FileNotFoundException)
            {
                shm = MemoryMappedFile.CreateOrOpen(shmName, TOTAL_SIZE);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Created shared memory: {shmName}");
            }
        }

        internal static void WriteFlag(int flag) => accessor.Write(0x00, flag);
        internal static int ReadFlag() => accessor.ReadInt32(0x00);

        internal static float[] ReadAction()
        {
            float[] action = new float[6];
            int offset = HEADER_SIZE + OBS_SIZE;
            for (int i = 0; i < 6; i++)
            {
                action[i] = accessor.ReadSingle(offset + i * 4);
            }
            return action;
        }

        internal static void WriteReward(float reward) => accessor.Write(0x08, reward);
        internal static void WriteDone(bool done) => accessor.Write(0x0C, done ? 1 : 0);
        internal static void WriteStep(int step) => accessor.Write(0x10, step);
        internal static void WriteGameTime(int timeMs) => accessor.Write(0x14, timeMs);
        internal static void WriteObservation(byte[] pixels)
        {
            accessor.WriteArray(HEADER_SIZE, pixels, 0, pixels.Length);
        }

        internal static void UnlockAllContent()
        {
            Log.LogInfo("Unlocking all content via reflection...");
            int unlockCount = 0;

            var allSOs = Resources.FindObjectsOfTypeAll<ScriptableObject>();
            foreach (var so in allSOs)
            {
                var type = so.GetType();
                var fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                foreach (var field in fields)
                {
                    string name = field.Name.ToLower();
                    if (field.FieldType == typeof(bool) &&
                        (name.Contains("unlock") || name.Contains("available") || name.Contains("purchased") || name.Contains("enabled")))
                    {
                        try { field.SetValue(so, true); unlockCount++; } catch { }
                    }
                }
            }

            var managers = Resources.FindObjectsOfTypeAll<MonoBehaviour>();
            foreach (var manager in managers)
            {
                var type = manager.GetType();
                string typeName = type.Name.ToLower();
                if (typeName.Contains("manager") || typeName.Contains("progression") || typeName.Contains("save"))
                {
                    var methods = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    foreach (var method in methods)
                    {
                        string methodName = method.Name.ToLower();
                        if ((methodName.Contains("unlock") || methodName.Contains("enable")) && method.GetParameters().Length == 0)
                        {
                            try { method.Invoke(manager, null); unlockCount++; } catch { }
                        }
                    }
                }
            }

            Log.LogInfo($"Unlocked {unlockCount} items/methods");
        }
    }

    public class ResearchUpdater : MonoBehaviour
    {
        private RenderTexture obsRT;
        private Camera obsCamera;
        private int currentStep = 0;

        void Start()
        {
            obsRT = new RenderTexture(
                ResearchPluginIL2CPP.OBS_WIDTH,
                ResearchPluginIL2CPP.OBS_HEIGHT,
                24,
                RenderTextureFormat.ARGB32
            );
            obsRT.Create();

            SetupObservationCamera();
        }

        private void SetupObservationCamera()
        {
            Camera mainCam = Camera.main;
            if (mainCam == null) return;

            GameObject obsCamObj = new GameObject("ObservationCamera");
            obsCamera = obsCamObj.AddComponent<Camera>();
            obsCamera.CopyFrom(mainCam);
            obsCamera.targetTexture = obsRT;
            obsCamera.enabled = true;

            obsCamObj.transform.SetParent(mainCam.transform);
            obsCamObj.transform.localPosition = Vector3.zero;
            obsCamObj.transform.localRotation = Quaternion.identity;
        }

        private void CaptureObservation()
        {
            if (obsCamera == null || obsRT == null) return;
            AsyncGPUReadback.Request(obsRT, 0, TextureFormat.RGB24, OnReadbackComplete);
        }

        private void OnReadbackComplete(AsyncGPUReadbackRequest request)
        {
            if (request.hasError)
            {
                ResearchPluginIL2CPP.Log?.LogWarning("GPU readback failed");
                return;
            }
            var data = request.GetData<byte>();
            byte[] pixels = data.ToArray();

            ResearchPluginIL2CPP.WriteObservation(pixels);
            ResearchPluginIL2CPP.WriteStep(currentStep);
            ResearchPluginIL2CPP.WriteGameTime((int)(Time.timeSinceLevelLoad * 1000));
            ResearchPluginIL2CPP.WriteFlag(ResearchPluginIL2CPP.FLAG_OBS_READY);
        }

        void LateUpdate()
        {
            if (!ResearchPluginIL2CPP.deterministicMode) return;

            CaptureObservation();
            while (ResearchPluginIL2CPP.ReadFlag() != ResearchPluginIL2CPP.FLAG_ACTION_READY)
            {
                if (ResearchPluginIL2CPP.ReadFlag() == ResearchPluginIL2CPP.FLAG_RESET)
                {
                    HandleReset();
                    return;
                }
                Thread.Sleep(1);
            }

            float[] action = ResearchPluginIL2CPP.ReadAction();
            ApplyAction(action);

            currentStep++;
        }

        private void ApplyAction(float[] action)
        {
            // action[0:2] = movement, action[2:6] = buttons
            GameObject player = GameObject.FindWithTag("Player");
            if (player == null) return;
            // Hook real controller here when available.
        }

        private void HandleReset()
        {
            currentStep = 0;
            ResearchPluginIL2CPP.Log?.LogInfo("Reset requested");
        }
    }

    public static class PluginInfo
    {
        public const string PLUGIN_GUID = "com.megabonk.research";
        public const string PLUGIN_NAME = "Megabonk Research Infrastructure";
        public const string PLUGIN_VERSION = "1.0.0";
    }

    [HarmonyPatch]
    public static class SkipIntroPatches { }

    [HarmonyPatch(typeof(UnityEngine.Random), "InitState")]
    public static class DeterministicRandomPatch
    {
        static void Prefix(ref int seed)
        {
            string seedStr = Environment.GetEnvironmentVariable("MEGABONK_SEED");
            if (!string.IsNullOrEmpty(seedStr) && int.TryParse(seedStr, out int envSeed))
            {
                seed = envSeed;
            }
        }
    }
}

#endif // IL2CPP

