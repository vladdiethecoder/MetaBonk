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
        internal static FileStream shmFile;
        internal static bool useFileShm = false;

        // Flags/layout
        internal const int HEADER_SIZE = 0x20;
        internal const int ACTION_SIZE = 6 * 4;
        internal const int HEADER_OBS_WIDTH_OFFSET = 0x18;
        internal const int HEADER_OBS_HEIGHT_OFFSET = 0x1C;
        internal static int ObsWidth = 84;
        internal static int ObsHeight = 84;
        internal static int ObsSize = ObsWidth * ObsHeight * 3;
        internal static int TotalSize = HEADER_SIZE + ObsSize + ACTION_SIZE;
        internal static int? ObsCullingMask = null;

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

            ConfigureObservationFromEnv();
            InitSharedMemory();
            WriteObsShape();
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
                shmFile?.Dispose();
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
                string shmDir = Environment.GetEnvironmentVariable("MEGABONK_RESEARCH_SHM_DIR") ?? "";
                if (!string.IsNullOrWhiteSpace(shmDir))
                {
                    string path = Path.Combine(shmDir, shmName);
                    shmFile = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
                    if (shmFile.Length < TotalSize)
                    {
                        shmFile.SetLength(TotalSize);
                    }
                    useFileShm = true;
                    Log.LogInfo($"Created shared memory file: {path}");
                    return;
                }

                shm = MemoryMappedFile.OpenExisting(shmName);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Connected to shared memory: {shmName}");
            }
            catch (FileNotFoundException)
            {
                shm = MemoryMappedFile.CreateOrOpen(shmName, TotalSize);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Created shared memory: {shmName}");
            }
            catch (Exception ex)
            {
                Log.LogWarning($"Failed to open shared memory: {ex.Message}");
                throw;
            }
        }

        internal static void WriteFlag(int flag) => WriteInt32(0x00, flag);
        internal static int ReadFlag() => ReadInt32(0x00);

        internal static float[] ReadAction()
        {
            float[] action = new float[6];
            int offset = HEADER_SIZE + ObsSize;
            for (int i = 0; i < 6; i++)
            {
                action[i] = ReadSingle(offset + i * 4);
            }
            return action;
        }

        internal static void WriteReward(float reward) => WriteSingle(0x08, reward);
        internal static void WriteDone(bool done) => WriteInt32(0x0C, done ? 1 : 0);
        internal static void WriteStep(int step) => WriteInt32(0x10, step);
        internal static void WriteGameTime(int timeMs) => WriteInt32(0x14, timeMs);
        internal static void WriteObservation(byte[] pixels)
        {
            WriteBytes(HEADER_SIZE, pixels, pixels.Length);
        }

        private static void WriteInt32(long offset, int value)
        {
            if (useFileShm && shmFile != null)
            {
                byte[] buf = BitConverter.GetBytes(value);
                shmFile.Seek(offset, SeekOrigin.Begin);
                shmFile.Write(buf, 0, buf.Length);
                return;
            }
            accessor.Write(offset, value);
        }

        private static int ReadInt32(long offset)
        {
            if (useFileShm && shmFile != null)
            {
                byte[] buf = new byte[4];
                shmFile.Seek(offset, SeekOrigin.Begin);
                shmFile.Read(buf, 0, buf.Length);
                return BitConverter.ToInt32(buf, 0);
            }
            return accessor.ReadInt32(offset);
        }

        private static void WriteSingle(long offset, float value)
        {
            if (useFileShm && shmFile != null)
            {
                byte[] buf = BitConverter.GetBytes(value);
                shmFile.Seek(offset, SeekOrigin.Begin);
                shmFile.Write(buf, 0, buf.Length);
                return;
            }
            accessor.Write(offset, value);
        }

        private static float ReadSingle(long offset)
        {
            if (useFileShm && shmFile != null)
            {
                byte[] buf = new byte[4];
                shmFile.Seek(offset, SeekOrigin.Begin);
                shmFile.Read(buf, 0, buf.Length);
                return BitConverter.ToSingle(buf, 0);
            }
            return accessor.ReadSingle(offset);
        }

        private static void WriteBytes(long offset, byte[] data, int length)
        {
            if (useFileShm && shmFile != null)
            {
                shmFile.Seek(offset, SeekOrigin.Begin);
                shmFile.Write(data, 0, length);
                return;
            }
            accessor.WriteArray(offset, data, 0, length);
        }

        internal static void WriteObsShape()
        {
            if (!useFileShm && accessor == null) return;
            WriteInt32(HEADER_OBS_WIDTH_OFFSET, ObsWidth);
            WriteInt32(HEADER_OBS_HEIGHT_OFFSET, ObsHeight);
        }

        internal static void ConfigureObservationFromEnv()
        {
            ObsWidth = ReadEnvInt("MEGABONK_OBS_WIDTH", ObsWidth);
            ObsHeight = ReadEnvInt("MEGABONK_OBS_HEIGHT", ObsHeight);
            ObsCullingMask = ReadEnvIntNullable("MEGABONK_OBS_CULLING_MASK");
            ObsSize = ObsWidth * ObsHeight * 3;
            TotalSize = HEADER_SIZE + ObsSize + ACTION_SIZE;
        }

        internal static int ReadEnvInt(string name, int fallback)
        {
            string val = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(val) && int.TryParse(val, out int parsed) && parsed > 0)
            {
                return parsed;
            }
            return fallback;
        }

        internal static int? ReadEnvIntNullable(string name)
        {
            string val = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(val) && int.TryParse(val, out int parsed))
            {
                return parsed;
            }
            return null;
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
        private Texture2D obsTex;
        private Rect obsRect;
        private Camera obsCamera;
        private int currentStep = 0;
        private byte[] obsPixelBuffer;
        private bool warnedReadbackSize = false;
        private bool loggedLateUpdate = false;
        private bool loggedFirstObs = false;

        void Start()
        {
            DontDestroyOnLoad(gameObject);
            obsRT = new RenderTexture(
                ResearchPluginIL2CPP.ObsWidth,
                ResearchPluginIL2CPP.ObsHeight,
                24,
                RenderTextureFormat.ARGB32
            );
            obsRT.Create();
            obsTex = new Texture2D(
                ResearchPluginIL2CPP.ObsWidth,
                ResearchPluginIL2CPP.ObsHeight,
                TextureFormat.RGB24,
                false
            );
            obsRect = new Rect(0, 0, ResearchPluginIL2CPP.ObsWidth, ResearchPluginIL2CPP.ObsHeight);
            obsPixelBuffer = new byte[ResearchPluginIL2CPP.ObsSize];

            SetupObservationCamera();
        }

        private void SetupObservationCamera()
        {
            Camera mainCam = Camera.main;
            if (mainCam == null && Camera.allCamerasCount > 0)
            {
                Camera[] cams = new Camera[Camera.allCamerasCount];
                Camera.GetAllCameras(cams);
                if (cams.Length > 0)
                {
                    mainCam = cams[0];
                }
            }
            if (mainCam == null)
            {
                ResearchPluginIL2CPP.Log?.LogWarning("No camera found for observation capture.");
                return;
            }

            GameObject obsCamObj = new GameObject("ObservationCamera");
            obsCamera = obsCamObj.AddComponent<Camera>();
            obsCamera.CopyFrom(mainCam);
            obsCamera.targetTexture = obsRT;
            obsCamera.enabled = true;
            if (ResearchPluginIL2CPP.ObsCullingMask.HasValue)
            {
                obsCamera.cullingMask = ResearchPluginIL2CPP.ObsCullingMask.Value;
            }

            obsCamObj.transform.SetParent(mainCam.transform);
            obsCamObj.transform.localPosition = Vector3.zero;
            obsCamObj.transform.localRotation = Quaternion.identity;
        }

        private void CaptureObservation()
        {
            if (obsRT == null || !obsRT.IsCreated())
            {
                obsRT = new RenderTexture(
                    ResearchPluginIL2CPP.ObsWidth,
                    ResearchPluginIL2CPP.ObsHeight,
                    24,
                    RenderTextureFormat.ARGB32
                );
                obsRT.Create();
            }
            if (obsTex == null || obsTex.width != ResearchPluginIL2CPP.ObsWidth || obsTex.height != ResearchPluginIL2CPP.ObsHeight)
            {
                obsTex = new Texture2D(
                    ResearchPluginIL2CPP.ObsWidth,
                    ResearchPluginIL2CPP.ObsHeight,
                    TextureFormat.RGB24,
                    false
                );
                obsRect = new Rect(0, 0, ResearchPluginIL2CPP.ObsWidth, ResearchPluginIL2CPP.ObsHeight);
            }
            if (obsCamera == null)
            {
                SetupObservationCamera();
            }
            if (obsCamera == null || obsRT == null || obsTex == null) return;

            if (obsCamera.targetTexture != obsRT)
            {
                obsCamera.targetTexture = obsRT;
            }

            try
            {
                obsCamera.Render();
                RenderTexture prev = RenderTexture.active;
                RenderTexture.active = obsRT;
                obsTex.ReadPixels(obsRect, 0, 0);
                obsTex.Apply(false, false);
                RenderTexture.active = prev;

                var data = obsTex.GetRawTextureData<byte>();
                if (obsPixelBuffer == null || data.Length != obsPixelBuffer.Length)
                {
                    if (!warnedReadbackSize)
                    {
                        ResearchPluginIL2CPP.Log?.LogWarning(
                            $"Readback size mismatch: expected {obsPixelBuffer?.Length ?? 0} bytes, got {data.Length}"
                        );
                        warnedReadbackSize = true;
                    }
                    return;
                }
                var pixels = data.ToArray();
                System.Buffer.BlockCopy(pixels, 0, obsPixelBuffer, 0, obsPixelBuffer.Length);

                ResearchPluginIL2CPP.WriteObservation(obsPixelBuffer);
                ResearchPluginIL2CPP.WriteStep(currentStep);
                ResearchPluginIL2CPP.WriteGameTime((int)(Time.timeSinceLevelLoad * 1000));
                ResearchPluginIL2CPP.WriteFlag(ResearchPluginIL2CPP.FLAG_OBS_READY);
                ResearchPluginIL2CPP.accessor?.Flush();
                if (!loggedFirstObs)
                {
                    ResearchPluginIL2CPP.Log?.LogInfo("Wrote first observation frame.");
                    loggedFirstObs = true;
                }
            }
            catch (Exception ex)
            {
                ResearchPluginIL2CPP.Log?.LogWarning($"ReadPixels capture failed: {ex.Message}");
            }
        }

        void LateUpdate()
        {
            if (!loggedLateUpdate)
            {
                ResearchPluginIL2CPP.Log?.LogInfo("ResearchUpdater LateUpdate running.");
                loggedLateUpdate = true;
            }
            CaptureObservation();
            if (!ResearchPluginIL2CPP.deterministicMode) return;
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
