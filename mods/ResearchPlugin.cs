// BepInEx Research Mod for Megabonk (Mono build).
//
// NOTE: MegaBonk is IL2CPP. For IL2CPP/BepInEx 6 builds, use
// `ResearchPluginIL2CPP.cs` in the same folder. This file is excluded
// when the `IL2CPP` compilation constant is set.

#if !IL2CPP

// BepInEx Research Mod for Megabonk
// Provides research infrastructure capabilities:
// - Unlock all content via reflection
// - Deterministic stepping
// - Shared memory IPC with Python
// - AsyncGPUReadback observation export

// File: ResearchUnlocker.cs
// Deploy to: BepInEx/plugins/

using BepInEx;
using BepInEx.Logging;
using HarmonyLib;
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace MegabonkResearch
{
    [BepInPlugin(PluginInfo.PLUGIN_GUID, PluginInfo.PLUGIN_NAME, PluginInfo.PLUGIN_VERSION)]
    public class ResearchPlugin : BaseUnityPlugin
    {
        // ═══════════════════════════════════════════════════════════
        // CONFIGURATION
        // ═══════════════════════════════════════════════════════════
        
        private const int HEADER_SIZE = 0x20;
        private const int ACTION_SIZE = 6 * 4; // 6 floats
        private const int HEADER_OBS_WIDTH_OFFSET = 0x18;
        private const int HEADER_OBS_HEIGHT_OFFSET = 0x1C;
        
        // Flag values
        private const int FLAG_WAIT = 0;
        private const int FLAG_ACTION_READY = 1;
        private const int FLAG_OBS_READY = 2;
        private const int FLAG_RESET = 3;
        private const int FLAG_TERMINATED = 4;
        
        // ═══════════════════════════════════════════════════════════
        // STATE
        // ═══════════════════════════════════════════════════════════
        
        private static ManualLogSource Log;
        private static Harmony harmony;
        
        // Shared memory
        private MemoryMappedFile shm;
        private MemoryMappedViewAccessor accessor;
        private string shmName;
        private int obsWidth = 84;
        private int obsHeight = 84;
        private int obsSize;
        private int totalSize;
        private int? obsCullingMask;
        
        // Rendering
        private RenderTexture obsRT;
        private Camera obsCamera;
        private byte[] obsPixelBuffer;
        private bool readbackPending = false;
        private bool warnedReadbackSize = false;
        
        // Stepping
        private bool deterministicMode = false;
        private bool waitingForAction = false;
        private int currentStep = 0;
        
        // ═══════════════════════════════════════════════════════════
        // LIFECYCLE
        // ═══════════════════════════════════════════════════════════
        
        void Awake()
        {
            Log = Logger;
            Log.LogInfo("=== Megabonk Research Infrastructure ===");
            
            // Get instance ID from environment
            string instanceId = Environment.GetEnvironmentVariable("MEGABONK_INSTANCE_ID") ?? "0";
            shmName = $"megabonk_env_{instanceId}";
            
            // Check if deterministic mode is enabled
            deterministicMode = Environment.GetEnvironmentVariable("MEGABONK_DETERMINISTIC") == "1";
            
            // Configure observation shape before shared memory init.
            ConfigureObservationFromEnv();

            // Initialize shared memory
            InitSharedMemory();
            WriteObsShape();
            
            // Unlock all content
            UnlockAllContent();
            
            // Apply Harmony patches
            harmony = new Harmony(PluginInfo.PLUGIN_GUID);
            harmony.PatchAll();
            
            if (deterministicMode)
            {
                Log.LogInfo("Deterministic stepping enabled");
            }
            
            Log.LogInfo($"Research infrastructure initialized (instance: {instanceId})");
        }
        
        void OnDestroy()
        {
            accessor?.Dispose();
            shm?.Dispose();
            harmony?.UnpatchSelf();
        }
        
        // ═══════════════════════════════════════════════════════════
        // SHARED MEMORY
        // ═══════════════════════════════════════════════════════════
        
        private void InitSharedMemory()
        {
            try
            {
                string shmDir = Environment.GetEnvironmentVariable("MEGABONK_RESEARCH_SHM_DIR") ?? "";
                if (!string.IsNullOrWhiteSpace(shmDir))
                {
                    string path = Path.Combine(shmDir, shmName);
                    shm = MemoryMappedFile.CreateFromFile(path, FileMode.OpenOrCreate, null, totalSize);
                    accessor = shm.CreateViewAccessor();
                    Log.LogInfo($"Created shared memory file: {path}");
                    return;
                }

                // Try to open existing (Python creates it)
                shm = MemoryMappedFile.OpenExisting(shmName);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Connected to shared memory: {shmName}");
            }
            catch (FileNotFoundException)
            {
                // Create new if Python hasn't created it yet
                shm = MemoryMappedFile.CreateOrOpen(shmName, totalSize);
                accessor = shm.CreateViewAccessor();
                Log.LogInfo($"Created shared memory: {shmName}");
            }
            catch (Exception ex)
            {
                Log.LogWarning($"Failed to open shared memory: {ex.Message}");
                throw;
            }
        }
        
        private void WriteFlag(int flag)
        {
            accessor.Write(0x00, flag);
        }
        
        private int ReadFlag()
        {
            return accessor.ReadInt32(0x00);
        }
        
        private float[] ReadAction()
        {
            float[] action = new float[6];
            int offset = HEADER_SIZE + obsSize;
            
            for (int i = 0; i < 6; i++)
            {
                action[i] = accessor.ReadSingle(offset + i * 4);
            }
            
            return action;
        }
        
        private void WriteReward(float reward)
        {
            accessor.Write(0x08, reward);
        }
        
        private void WriteDone(bool done)
        {
            accessor.Write(0x0C, done ? 1 : 0);
        }
        
        private void WriteStep(int step)
        {
            accessor.Write(0x10, step);
        }
        
        private void WriteGameTime(int timeMs)
        {
            accessor.Write(0x14, timeMs);
        }
        
        private void WriteObservation(byte[] pixels)
        {
            accessor.WriteArray(HEADER_SIZE, pixels, 0, pixels.Length);
        }

        private void WriteObsShape()
        {
            if (accessor == null) return;
            accessor.Write(HEADER_OBS_WIDTH_OFFSET, obsWidth);
            accessor.Write(HEADER_OBS_HEIGHT_OFFSET, obsHeight);
        }
        
        // ═══════════════════════════════════════════════════════════
        // OBSERVATION CAPTURE
        // ═══════════════════════════════════════════════════════════
        
        void Start()
        {
            DontDestroyOnLoad(gameObject);
            // Create observation render texture
            obsRT = new RenderTexture(obsWidth, obsHeight, 24, RenderTextureFormat.ARGB32);
            obsRT.Create();
            obsPixelBuffer = new byte[obsSize];
            
            // Find or create observation camera
            SetupObservationCamera();
        }
        
        private void SetupObservationCamera()
        {
            // Try to find existing camera
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
                Log.LogWarning("No camera found for observation capture.");
                return;
            }

            // Create a child camera for observations
            GameObject obsCamObj = new GameObject("ObservationCamera");
            obsCamera = obsCamObj.AddComponent<Camera>();
            obsCamera.CopyFrom(mainCam);
            obsCamera.targetTexture = obsRT;
            obsCamera.enabled = true;
            if (obsCullingMask.HasValue)
            {
                obsCamera.cullingMask = obsCullingMask.Value;
            }

            // Follow main camera
            obsCamObj.transform.SetParent(mainCam.transform);
            obsCamObj.transform.localPosition = Vector3.zero;
            obsCamObj.transform.localRotation = Quaternion.identity;
        }
        
        private void CaptureObservation()
        {
            if (obsCamera == null)
            {
                SetupObservationCamera();
            }
            if (obsCamera == null || obsRT == null || readbackPending) return;

            // Use AsyncGPUReadback for non-blocking capture
            readbackPending = true;
            AsyncGPUReadback.Request(obsRT, 0, TextureFormat.RGB24, OnReadbackComplete);
        }
        
        private void OnReadbackComplete(AsyncGPUReadbackRequest request)
        {
            readbackPending = false;
            if (request.hasError)
            {
                Log.LogWarning("GPU readback failed");
                return;
            }

            var data = request.GetData<byte>();
            if (obsPixelBuffer == null || data.Length != obsPixelBuffer.Length)
            {
                if (!warnedReadbackSize)
                {
                    Log.LogWarning($"Readback size mismatch: expected {obsPixelBuffer?.Length ?? 0} bytes, got {data.Length}");
                    warnedReadbackSize = true;
                }
                return;
            }
            data.CopyTo(obsPixelBuffer);
            
            // Write to shared memory
            WriteObservation(obsPixelBuffer);
            WriteStep(currentStep);
            WriteGameTime((int)(Time.timeSinceLevelLoad * 1000));
            
            // Signal observation ready
            WriteFlag(FLAG_OBS_READY);
        }
        
        // ═══════════════════════════════════════════════════════════
        // DETERMINISTIC STEPPING
        // ═══════════════════════════════════════════════════════════
        
        void LateUpdate()
        {
            // Capture observation
            CaptureObservation();
            if (!deterministicMode) return;
            
            // Wait for action
            waitingForAction = true;
            while (waitingForAction && ReadFlag() != FLAG_ACTION_READY)
            {
                Thread.Sleep(1); // Yield CPU
                
                // Check for reset
                if (ReadFlag() == FLAG_RESET)
                {
                    HandleReset();
                    return;
                }
            }
            
            // Read action and apply
            float[] action = ReadAction();
            ApplyAction(action);
            
            waitingForAction = false;
            currentStep++;
        }
        
        private void ApplyAction(float[] action)
        {
            // action[0:2] = movement (x, y)
            // action[2:6] = buttons
            
            // Find player and apply movement
            GameObject player = GameObject.FindWithTag("Player");
            if (player != null)
            {
                // Would inject into player controller here
                // This depends on Megabonk's specific implementation
            }
        }

        private void ConfigureObservationFromEnv()
        {
            obsWidth = ReadEnvInt("MEGABONK_OBS_WIDTH", obsWidth);
            obsHeight = ReadEnvInt("MEGABONK_OBS_HEIGHT", obsHeight);
            obsCullingMask = ReadEnvIntNullable("MEGABONK_OBS_CULLING_MASK");
            obsSize = obsWidth * obsHeight * 3;
            totalSize = HEADER_SIZE + obsSize + ACTION_SIZE;
        }

        private static int ReadEnvInt(string name, int fallback)
        {
            string val = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(val) && int.TryParse(val, out int parsed) && parsed > 0)
            {
                return parsed;
            }
            return fallback;
        }

        private static int? ReadEnvIntNullable(string name)
        {
            string val = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(val) && int.TryParse(val, out int parsed))
            {
                return parsed;
            }
            return null;
        }
        
        private void HandleReset()
        {
            // Reset game state
            currentStep = 0;
            
            // Would trigger game restart here
            // SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            
            Log.LogInfo("Reset requested");
        }
        
        // ═══════════════════════════════════════════════════════════
        // CONTENT UNLOCKING
        // ═══════════════════════════════════════════════════════════
        
        private void UnlockAllContent()
        {
            Log.LogInfo("Unlocking all content via reflection...");
            
            int unlockCount = 0;
            
            // Find all ScriptableObjects
            var allSOs = Resources.FindObjectsOfTypeAll<ScriptableObject>();
            
            foreach (var so in allSOs)
            {
                var type = so.GetType();
                var fields = type.GetFields(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                );
                
                foreach (var field in fields)
                {
                    string name = field.Name.ToLower();
                    
                    // Look for unlock-related booleans
                    if (field.FieldType == typeof(bool) &&
                        (name.Contains("unlock") ||
                         name.Contains("available") ||
                         name.Contains("purchased") ||
                         name.Contains("enabled")))
                    {
                        try
                        {
                            field.SetValue(so, true);
                            unlockCount++;
                        }
                        catch { }
                    }
                }
            }
            
            // Find singleton managers
            var managers = FindObjectsOfType<MonoBehaviour>();
            
            foreach (var manager in managers)
            {
                var type = manager.GetType();
                string typeName = type.Name.ToLower();
                
                if (typeName.Contains("manager") ||
                    typeName.Contains("progression") ||
                    typeName.Contains("save"))
                {
                    // Try to invoke unlock methods
                    var methods = type.GetMethods(
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                    );
                    
                    foreach (var method in methods)
                    {
                        string methodName = method.Name.ToLower();
                        if ((methodName.Contains("unlock") || methodName.Contains("enable")) &&
                            method.GetParameters().Length == 0)
                        {
                            try
                            {
                                method.Invoke(manager, null);
                                unlockCount++;
                            }
                            catch { }
                        }
                    }
                }
            }
            
            Log.LogInfo($"Unlocked {unlockCount} items/methods");
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // PLUGIN INFO
    // ═══════════════════════════════════════════════════════════════════
    
    public static class PluginInfo
    {
        public const string PLUGIN_GUID = "com.megabonk.research";
        public const string PLUGIN_NAME = "Megabonk Research Infrastructure";
        public const string PLUGIN_VERSION = "1.0.0";
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // HARMONY PATCHES
    // ═══════════════════════════════════════════════════════════════════
    
    // Patch to disable intro/splash screens
    [HarmonyPatch]
    public static class SkipIntroPatches
    {
        // Would patch specific intro methods here
        // Target depends on Megabonk's implementation
    }
    
    // Patch for deterministic random seed
    [HarmonyPatch(typeof(UnityEngine.Random), "InitState")]
    public static class DeterministicRandomPatch
    {
        static void Prefix(ref int seed)
        {
            // Use environment variable for seed
            string seedStr = Environment.GetEnvironmentVariable("MEGABONK_SEED");
            if (!string.IsNullOrEmpty(seedStr) && int.TryParse(seedStr, out int envSeed))
            {
                seed = envSeed;
            }
        }
    }
}

#endif // !IL2CPP
