// =============================================================================
// BonkLink.cs - BepInEx 6 Plugin for MetaBonk AI Bridge
// =============================================================================
// 
// Installation:
// 1. Download BepInEx 6.0 IL2CPP from: https://builds.bepinex.dev/projects/bepinex_be
// 2. Extract to MegaBonk game directory
// 3. Copy BonkLink.dll to BepInEx/plugins/
// 4. Run game - Python client can now connect
//
// =============================================================================

using BepInEx;
using BepInEx.Unity.IL2CPP;
using BepInEx.Configuration;
using BepInEx.Logging;
using HarmonyLib;
using Il2CppInterop.Runtime.Injection;
using System;
using System.IO;
using System.IO.Pipes;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using UnityEngine;

namespace BonkLink
{
    [BepInPlugin(GUID, NAME, VERSION)]
    public class BonkLinkPlugin : BasePlugin
    {
        public const string GUID = "com.metabonk.bonklink";
        public const string NAME = "BonkLink AI Bridge";
        public const string VERSION = "1.0.0";

        internal static new ManualLogSource Log;
        internal static BonkLinkPlugin Instance;
        
        // Configuration
        private ConfigEntry<int> _port;
        private ConfigEntry<bool> _useNamedPipe;
        private ConfigEntry<string> _pipeName;
        private ConfigEntry<int> _updateHz;
        private ConfigEntry<bool> _enableJpeg;
        private ConfigEntry<int> _jpegHz;
        private ConfigEntry<int> _jpegWidth;
        private ConfigEntry<int> _jpegHeight;
        private ConfigEntry<int> _jpegQuality;
        private ConfigEntry<bool> _enableInputSnapshot;
        private ConfigEntry<string> _frameFormat;
        
        // Networking
        private TcpListener _tcpListener;
        private NamedPipeServerStream _pipeServer;
        private NetworkStream _clientStream;
        private BinaryWriter _writer;
        private BinaryReader _reader;
        private Thread _serverThread;
        private Thread _updateThread;
        private bool _running;
        private bool _clientConnected;
        
        // State buffer
        private GameStateBuffer _stateBuffer;
        private ActionBuffer _actionBuffer;
        private object _stateLock = new object();
        private object _actionLock = new object();

        // Optional human input snapshot (disabled by default).
        private UnityInputSnapshot _hinp;
        
        // Frame capture
        private Texture2D _captureTexture;
        private byte[] _frameBuffer;
        private RenderTexture _captureRt;
        private bool _loggedFirstJpeg;

        private static void AttachIl2CppThread()
        {
            try
            {
                // BepInEx IL2CPP: managed threads must be attached to the IL2CPP domain
                // before touching any IL2CPP-backed objects (ConfigEntry, logging, Unity types, etc).
                Il2CppInterop.Runtime.IL2CPP.il2cpp_thread_attach(Il2CppInterop.Runtime.IL2CPP.il2cpp_domain_get());
            }
            catch { }
        }

        private static bool IsWine()
        {
            try
            {
                // Wine/Proton commonly sets at least one of these.
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WINEPREFIX"))) return true;
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WINELOADER"))) return true;
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WINELOADERNOEXEC"))) return true;
            }
            catch { }
            return false;
        }

        public override void Load()
        {
            Instance = this;
            Log = base.Log;
            Log.LogInfo("BonkLink loading...");
            
            // Configuration
            _port = Config.Bind("Network", "Port", 5555, "TCP port for Python client");
            _useNamedPipe = Config.Bind("Network", "UseNamedPipe", false, "Use Named Pipe instead of TCP");
            _pipeName = Config.Bind("Network", "PipeName", "BonkLink", "Named pipe name");
            _updateHz = Config.Bind("Performance", "UpdateHz", 60, "State updates per second");
            _enableJpeg = Config.Bind("Capture", "EnableJpeg", true, "Send a JPEG frame along with state each tick");
            _jpegHz = Config.Bind("Capture", "JpegHz", 10, "JPEG capture rate (frames per second)");
            _jpegWidth = Config.Bind("Capture", "JpegWidth", 320, "JPEG frame width");
            _jpegHeight = Config.Bind("Capture", "JpegHeight", 180, "JPEG frame height");
            _jpegQuality = Config.Bind("Capture", "JpegQuality", 75, "JPEG quality (1-100)");
            _enableInputSnapshot = Config.Bind("Capture", "EnableInputSnapshot", false, "Append a tagged input snapshot to state (off by default).");
            // Unity's EncodeToJPG can crash under Wine/Proton (native AccessViolation). Default to raw frames there.
            var defaultFmt = IsWine() ? "raw_rgb" : "jpeg";
            _frameFormat = Config.Bind(
                "Capture",
                "FrameFormat",
                defaultFmt,
                "Frame payload encoding: 'jpeg' (Unity EncodeToJPG; may crash under Wine/Proton) or 'raw_rgb' (uncompressed RGB with MBRF header)."
            );
            
            // Initialize buffers
            _stateBuffer = new GameStateBuffer();
            _actionBuffer = new ActionBuffer();
            
            // Start server thread
            _running = true;
            _serverThread = new Thread(ServerLoop);
            _serverThread.IsBackground = true;
            _serverThread.Start();
            
            // Start update thread (replaces MonoBehaviour Update)
            _updateThread = new Thread(UpdateLoop);
            _updateThread.IsBackground = true;
            _updateThread.Start();

            // Main-thread updater (required for safe frame capture / Unity API access).
            try
            {
                try
                {
                    ClassInjector.RegisterTypeInIl2Cpp<BonkLinkUpdater>();
                }
                catch (Exception e)
                {
                    Log.LogWarning($"BonkLinkUpdater type registration failed (may already be registered): {e.Message}");
                }
                var go = new GameObject("BonkLinkUpdater");
                UnityEngine.Object.DontDestroyOnLoad(go);
                go.hideFlags = HideFlags.HideAndDontSave;
                go.AddComponent<BonkLinkUpdater>();
            }
            catch (Exception e)
            {
                Log.LogWarning($"Failed to create BonkLinkUpdater: {e.Message}");
            }

            // Register hooks
            Harmony.CreateAndPatchAll(typeof(GameHooks));
            
            Log.LogInfo($"BonkLink started on port {_port.Value}");
        }

        private void UpdateLoop()
        {
            AttachIl2CppThread();
            var hz = Math.Max(1, _updateHz.Value);
            var interval = TimeSpan.FromMilliseconds(Math.Max(1.0, 1000.0 / hz));
            while (_running)
            {
                try
                {
                    ReceiveAction();
                    SendState();
                    Thread.Sleep(interval);
                }
                catch (Exception e)
                {
                    Log.LogWarning($"Update loop error: {e.Message}");
                    Thread.Sleep(100);
                }
            }
        }

        private void ServerLoop()
        {
            AttachIl2CppThread();
            try
            {
                if (_useNamedPipe.Value)
                {
                    RunNamedPipeServer();
                }
                else
                {
                    RunTcpServer();
                }
            }
            catch (Exception e)
            {
                Log.LogError($"Server error: {e}");
            }
        }

        private void RunTcpServer()
        {
            _tcpListener = new TcpListener(IPAddress.Loopback, _port.Value);
            _tcpListener.Start();
            Log.LogInfo($"TCP server listening on localhost:{_port.Value}");

            while (_running)
            {
                try
                {
                    if (_tcpListener.Pending())
                    {
                        var client = _tcpListener.AcceptTcpClient();
                        client.NoDelay = true; // Disable Nagle for low latency
                        _clientStream = client.GetStream();
                        _writer = new BinaryWriter(_clientStream);
                        _reader = new BinaryReader(_clientStream);
                        _clientConnected = true;
                        Log.LogInfo("Python client connected");
                    }
                    Thread.Sleep(10);
                }
                catch (Exception e)
                {
                    if (_running) Log.LogWarning($"TCP accept error: {e.Message}");
                }
            }
        }

        private void RunNamedPipeServer()
        {
            while (_running)
            {
                try
                {
                    _pipeServer = new NamedPipeServerStream(
                        _pipeName.Value,
                        PipeDirection.InOut,
                        1,
                        PipeTransmissionMode.Byte,
                        PipeOptions.Asynchronous
                    );
                    
                    Log.LogInfo($"Named pipe '{_pipeName.Value}' waiting for connection...");
                    _pipeServer.WaitForConnection();
                    
                    _writer = new BinaryWriter(_pipeServer);
                    _reader = new BinaryReader(_pipeServer);
                    _clientConnected = true;
                    Log.LogInfo("Python client connected via Named Pipe");
                    
                    // Wait until disconnected
                    while (_pipeServer.IsConnected && _running)
                    {
                        Thread.Sleep(100);
                    }
                    
                    _clientConnected = false;
                    _pipeServer.Dispose();
                }
                catch (Exception e)
                {
                    if (_running) Log.LogWarning($"Pipe error: {e.Message}");
                    Thread.Sleep(1000);
                }
            }
        }

        /// <summary>
        /// Send current game state to Python client
        /// </summary>
        public void SendState()
        {
            if (!_clientConnected || _writer == null) return;

            try
            {
                byte[] stateData;
                byte[] frameData;
                lock (_stateLock)
                {
                    // Serialize state to bytes
                    stateData = _stateBuffer.Serialize();
                    // Send at most one frame per capture (avoid re-sending the same frame every tick,
                    // which is especially expensive for raw RGB frames).
                    frameData = _frameBuffer;
                    _frameBuffer = null;
                }

                // Write header: state_size (4 bytes)
                _writer.Write(stateData.Length);
                // Write state data
                _writer.Write(stateData);

                // Write frame if available
                if (frameData != null && frameData.Length > 0)
                {
                    _writer.Write(frameData.Length);
                    _writer.Write(frameData);
                }
                else
                {
                    _writer.Write(0);
                }

                _writer.Flush();
            }
            catch (Exception e)
            {
                Log.LogWarning($"Send error: {e.Message}");
                _clientConnected = false;
            }
        }

        /// <summary>
        /// Receive action from Python client
        /// </summary>
        public void ReceiveAction()
        {
            if (!_clientConnected || _reader == null) return;
            if (_clientStream != null && !_clientStream.DataAvailable) return;

            try
            {
                lock (_actionLock)
                {
                    int actionSize = _reader.ReadInt32();
                    if (actionSize > 0 && actionSize < 1024)
                    {
                        byte[] actionData = _reader.ReadBytes(actionSize);
                        _actionBuffer.Deserialize(actionData);
                    }
                }
            }
            catch (Exception e)
            {
                Log.LogWarning($"Receive error: {e.Message}");
            }
        }

        /// <summary>
        /// Update state buffer with current game data
        /// </summary>
        public void UpdateStateBuffer()
        {
            lock (_stateLock)
            {
                // Player state (hook these to actual game objects)
                _stateBuffer.playerPosition = GetPlayerPosition();
                _stateBuffer.playerVelocity = GetPlayerVelocity();
                _stateBuffer.playerHealth = GetPlayerHealth();
                _stateBuffer.playerMaxHealth = GetPlayerMaxHealth();
                
                // Game state
                _stateBuffer.gameTime = Time.time;
                _stateBuffer.isPlaying = IsInGameplay();
                _stateBuffer.isPaused = IsPaused();
                
                // Enemies (first 10)
                _stateBuffer.enemies = GetNearbyEnemies(10);
                
                // Items/Pickups
                _stateBuffer.pickups = GetNearbyPickups(5);
                
                // UI state
                _stateBuffer.currentMenu = GetCurrentMenu();
                _stateBuffer.levelUpOptions = GetLevelUpOptions();

                // Optional input snapshot trailer (off by default; does not affect visual-only learning).
                bool wantInput = _enableInputSnapshot != null && _enableInputSnapshot.Value;
                _stateBuffer.includeInputTrailer = wantInput && _hinp != null;
                if (_stateBuffer.includeInputTrailer)
                {
                    _stateBuffer.inputMoveX = _hinp.moveX;
                    _stateBuffer.inputMoveY = _hinp.moveY;
                    _stateBuffer.inputLookX = _hinp.lookX;
                    _stateBuffer.inputLookY = _hinp.lookY;
                    _stateBuffer.inputFire = _hinp.fire;
                    _stateBuffer.inputAbility = _hinp.ability;
                    _stateBuffer.inputInteract = _hinp.interact;
                    _stateBuffer.inputUiClick = _hinp.uiClick;
                    _stateBuffer.inputClickNX = _hinp.clickNX;
                    _stateBuffer.inputClickNY = _hinp.clickNY;
                }
            }
        }

        internal void UpdateHumanInputSnapshot()
        {
            // Main-thread only. Disabled by default.
            try
            {
                if (_enableInputSnapshot == null || !_enableInputSnapshot.Value)
                    return;
            }
            catch { return; }

            try
            {
                var snap = UnityInputSnapshot.Capture();
                if (snap == null) return;
                lock (_stateLock)
                {
                    _hinp = snap;
                }
            }
            catch { }
        }

        /// <summary>
        /// Apply received action to game
        /// </summary>
        public void ApplyAction()
        {
            lock (_actionLock)
            {
                if (!_actionBuffer.hasAction) return;
                
                // Movement
                SetPlayerInput(
                    _actionBuffer.moveX,
                    _actionBuffer.moveY
                );
                
                // Aim/Look
                SetPlayerAim(
                    _actionBuffer.aimX,
                    _actionBuffer.aimY
                );
                
                // Actions
                if (_actionBuffer.fire) TriggerFire();
                if (_actionBuffer.ability) TriggerAbility();
                if (_actionBuffer.interact) TriggerInteract();
                
                // UI
                if (_actionBuffer.uiClick)
                {
                    ClickAtPosition(_actionBuffer.clickX, _actionBuffer.clickY);
                }
                
                _actionBuffer.hasAction = false;
            }
        }

        internal bool CaptureEnabled()
        {
            return _enableJpeg != null && _enableJpeg.Value;
        }

        internal int CaptureHz()
        {
            if (_jpegHz == null) return 0;
            return Math.Max(0, _jpegHz.Value);
        }

        internal int UpdateHz()
        {
            if (_updateHz == null) return 0;
            return Math.Max(0, _updateHz.Value);
        }

        public void CaptureFrame()
        {
            if (!_clientConnected) return;
            if (!CaptureEnabled()) return;
            
            try
            {
                int w = Math.Max(64, _jpegWidth.Value);
                int h = Math.Max(64, _jpegHeight.Value);
                int quality = Math.Max(1, Math.Min(100, _jpegQuality.Value));
                var fmt = "jpeg";
                try { fmt = (_frameFormat != null ? (_frameFormat.Value ?? "jpeg") : "jpeg").Trim().ToLowerInvariant(); } catch { fmt = "jpeg"; }

                if (_captureTexture == null || _captureTexture.width != w || _captureTexture.height != h)
                {
                    _captureTexture = new Texture2D(w, h, TextureFormat.RGB24, false);
                }

                if (_captureRt == null || _captureRt.width != w || _captureRt.height != h)
                {
                    try
                    {
                        _captureRt?.Release();
                    }
                    catch { }
                    _captureRt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
                    _captureRt.Create();
                }

                var src = ScreenCapture.CaptureScreenshotAsTexture();
                Graphics.Blit(src, _captureRt);
                var prev = RenderTexture.active;
                RenderTexture.active = _captureRt;
                _captureTexture.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
                _captureTexture.Apply(false, false);
                RenderTexture.active = prev;
                if (src != null) UnityEngine.Object.Destroy(src);

                if (fmt.StartsWith("raw"))
                {
                    // Raw RGB frame: [4 bytes 'MBRF'][int32 w][int32 h][int32 c][payload bytes]
                    // c=3, payload is row-major RGB.
                    var pixels = _captureTexture.GetPixels32();
                    int payloadLen = Math.Max(0, pixels.Length) * 3;
                    var frame = new byte[16 + payloadLen];
                    frame[0] = (byte)'M';
                    frame[1] = (byte)'B';
                    frame[2] = (byte)'R';
                    frame[3] = (byte)'F';
                    Buffer.BlockCopy(BitConverter.GetBytes(w), 0, frame, 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(h), 0, frame, 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(3), 0, frame, 12, 4);
                    int o = 16;
                    for (int i = 0; i < pixels.Length; i++)
                    {
                        var p = pixels[i];
                        frame[o++] = p.r;
                        frame[o++] = p.g;
                        frame[o++] = p.b;
                    }
                    lock (_stateLock)
                    {
                        _frameBuffer = frame;
                    }
                    if (!_loggedFirstJpeg && frame.Length > 16)
                    {
                        _loggedFirstJpeg = true;
                        Log.LogInfo($"Raw RGB capture active ({w}x{h} @ ~{CaptureHz()}Hz); first frame {frame.Length} bytes");
                    }
                }
                else
                {
                    // WARNING: EncodeToJPG is known to crash under Wine/Proton on some setups.
                    var jpg = ImageConversion.EncodeToJPG(_captureTexture, quality);
                    lock (_stateLock)
                    {
                        _frameBuffer = jpg;
                    }
                    if (!_loggedFirstJpeg && jpg != null && jpg.Length > 0)
                    {
                        _loggedFirstJpeg = true;
                        Log.LogInfo($"JPEG capture active ({w}x{h} @ ~{CaptureHz()}Hz, q={quality}); first frame {jpg.Length} bytes");
                    }
                }
            }
            catch (Exception e)
            {
                Log.LogWarning($"Frame capture error: {e.Message}");
            }
        }

        // =========================================================================
        // Game Access Methods (implement via Harmony hooks or reflection)
        // =========================================================================
        
        private Vector3 GetPlayerPosition()
        {
            // TODO: Hook to actual player transform
            // Example: return Player.Instance?.transform.position ?? Vector3.zero;
            return Vector3.zero;
        }

        private Vector3 GetPlayerVelocity()
        {
            // TODO: Hook to player rigidbody
            return Vector3.zero;
        }

        private float GetPlayerHealth() => 100f;
        private float GetPlayerMaxHealth() => 100f;
        private bool IsInGameplay() => true;
        private bool IsPaused() => false;
        
        private List<EnemyState> GetNearbyEnemies(int max) => new List<EnemyState>();
        private List<PickupState> GetNearbyPickups(int max) => new List<PickupState>();
        private string GetCurrentMenu() => "None";
        private List<string> GetLevelUpOptions() => new List<string>();
        
        private void SetPlayerInput(float x, float y) { }
        private void SetPlayerAim(float x, float y) { }
        private void TriggerFire() { }
        private void TriggerAbility() { }
        private void TriggerInteract() { }
        private void ClickAtPosition(int x, int y) { }
    }

    public class BonkLinkUpdater : MonoBehaviour
    {
        private float _nextCaptureTs;
        private float _nextTickTs;

        public BonkLinkUpdater(IntPtr ptr) : base(ptr) { }

        private void Update()
        {
            var inst = BonkLinkPlugin.Instance;
            if (inst == null) return;

            // Always refresh input snapshot on the main thread.
            inst.UpdateHumanInputSnapshot();

            // Main-thread tick for Unity APIs (state read + action application).
            // Unity frequently throws "called from unknown thread" errors when
            // Time/Input/ScreenCapture/etc are accessed off the main thread.
            int tickHz = inst.UpdateHz();
            if (tickHz > 0)
            {
                float nowTick = Time.unscaledTime;
                if (nowTick >= _nextTickTs)
                {
                    _nextTickTs = nowTick + (1.0f / Mathf.Max(1, tickHz));
                    inst.UpdateStateBuffer();
                    inst.ApplyAction();
                }
            }

            int hz = inst.CaptureHz();
            if (hz <= 0) return;
            if (!inst.CaptureEnabled()) return;

            float now = Time.unscaledTime;
            if (now < _nextCaptureTs) return;
            _nextCaptureTs = now + (1.0f / Mathf.Max(1, hz));
            inst.CaptureFrame();
        }
    }

    /// <summary>
    /// Harmony hooks for game internals
    /// </summary>
    [HarmonyPatch]
    public static class GameHooks
    {
        // Example: Hook player damage
        // [HarmonyPatch(typeof(PlayerHealth), "TakeDamage")]
        // [HarmonyPostfix]
        // public static void OnPlayerDamage(float damage)
        // {
        //     BonkLinkPlugin.Log.LogInfo($"Player took {damage} damage");
        // }
    }

    // =========================================================================
    // Data Structures
    // =========================================================================

    [Serializable]
    public struct EnemyState
    {
        public Vector3 position;
        public Vector3 velocity;
        public float health;
        public string type;
    }

    [Serializable]
    public struct PickupState
    {
        public Vector3 position;
        public string type;
        public float value;
    }

    public class GameStateBuffer
    {
        // Player
        public Vector3 playerPosition;
        public Vector3 playerVelocity;
        public float playerHealth;
        public float playerMaxHealth;
        
        // Game
        public float gameTime;
        public bool isPlaying;
        public bool isPaused;
        
        // Entities
        public List<EnemyState> enemies = new List<EnemyState>();
        public List<PickupState> pickups = new List<PickupState>();
        
        // UI
        public string currentMenu;
        public List<string> levelUpOptions = new List<string>();

        // Optional human input snapshot (appended with a tagged trailer).
        public bool includeInputTrailer;
        public float inputMoveX;
        public float inputMoveY;
        public float inputLookX;
        public float inputLookY;
        public bool inputFire;
        public bool inputAbility;
        public bool inputInteract;
        public bool inputUiClick;
        // Normalized click coords in [0,1], origin top-left, relative to current Screen width/height.
        public float inputClickNX;
        public float inputClickNY;

        public byte[] Serialize()
        {
            using (var ms = new MemoryStream())
            using (var bw = new BinaryWriter(ms))
            {
                // Player
                bw.Write(playerPosition.x);
                bw.Write(playerPosition.y);
                bw.Write(playerPosition.z);
                bw.Write(playerVelocity.x);
                bw.Write(playerVelocity.y);
                bw.Write(playerVelocity.z);
                bw.Write(playerHealth);
                bw.Write(playerMaxHealth);
                
                // Game
                bw.Write(gameTime);
                bw.Write(isPlaying);
                bw.Write(isPaused);
                
                // Enemies
                bw.Write(enemies.Count);
                foreach (var e in enemies)
                {
                    bw.Write(e.position.x);
                    bw.Write(e.position.y);
                    bw.Write(e.position.z);
                    bw.Write(e.health);
                }
                
                // Menu
                bw.Write(currentMenu ?? "");
                bw.Write(levelUpOptions.Count);
                foreach (var opt in levelUpOptions)
                {
                    bw.Write(opt ?? "");
                }

                if (includeInputTrailer)
                {
                    // Tagged optional trailer so Python can parse input without breaking older clients.
                    // Magic = 'HINP' (little-endian), version=1.
                    bw.Write(0x504E4948u);
                    bw.Write(1);
                    bw.Write(inputMoveX);
                    bw.Write(inputMoveY);
                    bw.Write(inputLookX);
                    bw.Write(inputLookY);
                    bw.Write(inputFire);
                    bw.Write(inputAbility);
                    bw.Write(inputInteract);
                    bw.Write(inputUiClick);
                    bw.Write(inputClickNX);
                    bw.Write(inputClickNY);
                }
                
                return ms.ToArray();
            }
        }
    }

    public class ActionBuffer
    {
        public bool hasAction;
        
        // Movement (-1 to 1)
        public float moveX;
        public float moveY;
        
        // Aim direction
        public float aimX;
        public float aimY;
        
        // Discrete actions
        public bool fire;
        public bool ability;
        public bool interact;
        
        // UI interaction
        public bool uiClick;
        public int clickX;
        public int clickY;

        public void Deserialize(byte[] data)
        {
            using (var ms = new MemoryStream(data))
            using (var br = new BinaryReader(ms))
            {
                moveX = br.ReadSingle();
                moveY = br.ReadSingle();
                aimX = br.ReadSingle();
                aimY = br.ReadSingle();
                fire = br.ReadBoolean();
                ability = br.ReadBoolean();
                interact = br.ReadBoolean();
                uiClick = br.ReadBoolean();
                if (uiClick)
                {
                    clickX = br.ReadInt32();
                    clickY = br.ReadInt32();
                }
                hasAction = true;
            }
        }
    }

    internal class UnityInputSnapshot
    {
        public float moveX;
        public float moveY;
        public float lookX;
        public float lookY;
        public bool fire;
        public bool ability;
        public bool interact;
        public bool uiClick;
        public float clickNX;
        public float clickNY;

        // Reflection-based capture to avoid compile-time dependency on UnityEngine.Input modules.
        private static Type _inputType;
        private static Type _keyCodeType;
        private static System.Reflection.MethodInfo _getAxisRaw;
        private static System.Reflection.MethodInfo _getMouseButton;
        private static System.Reflection.MethodInfo _getMouseButtonDown;
        private static System.Reflection.MethodInfo _getKey;
        private static System.Reflection.PropertyInfo _mousePosition;

        private static void Ensure()
        {
            if (_inputType != null) return;
            _inputType = Type.GetType("UnityEngine.Input, UnityEngine.InputLegacyModule")
                ?? Type.GetType("UnityEngine.Input, UnityEngine.InputModule");
            if (_inputType == null) return;
            _keyCodeType = Type.GetType("UnityEngine.KeyCode, UnityEngine.CoreModule");
            try { _getAxisRaw = _inputType.GetMethod("GetAxisRaw", new[] { typeof(string) }); } catch { }
            try { _getMouseButton = _inputType.GetMethod("GetMouseButton", new[] { typeof(int) }); } catch { }
            try { _getMouseButtonDown = _inputType.GetMethod("GetMouseButtonDown", new[] { typeof(int) }); } catch { }
            try { _getKey = (_keyCodeType != null) ? _inputType.GetMethod("GetKey", new[] { _keyCodeType }) : null; } catch { }
            try { _mousePosition = _inputType.GetProperty("mousePosition"); } catch { }
        }

        private static float Axis(string name)
        {
            Ensure();
            if (_inputType == null || _getAxisRaw == null) return 0f;
            try { return (float)_getAxisRaw.Invoke(null, new object[] { name }); } catch { return 0f; }
        }

        private static bool MouseButton(int idx)
        {
            Ensure();
            if (_inputType == null || _getMouseButton == null) return false;
            try { return (bool)_getMouseButton.Invoke(null, new object[] { idx }); } catch { return false; }
        }

        private static bool MouseButtonDown(int idx)
        {
            Ensure();
            if (_inputType == null || _getMouseButtonDown == null) return false;
            try { return (bool)_getMouseButtonDown.Invoke(null, new object[] { idx }); } catch { return false; }
        }

        private static bool Key(string keyName)
        {
            Ensure();
            if (_inputType == null || _getKey == null || _keyCodeType == null) return false;
            try
            {
                var kc = Enum.Parse(_keyCodeType, keyName);
                return (bool)_getKey.Invoke(null, new object[] { kc });
            }
            catch { return false; }
        }

        private static Vector3 MousePos()
        {
            Ensure();
            if (_inputType == null || _mousePosition == null) return Vector3.zero;
            try { return (Vector3)_mousePosition.GetValue(null, null); } catch { return Vector3.zero; }
        }

        public static UnityInputSnapshot Capture()
        {
            Ensure();
            if (_inputType == null) return null;

            var s = new UnityInputSnapshot();
            s.moveX = Mathf.Clamp(Axis("Horizontal"), -1f, 1f);
            s.moveY = Mathf.Clamp(Axis("Vertical"), -1f, 1f);
            s.lookX = Mathf.Clamp(Axis("Mouse X"), -1f, 1f);
            s.lookY = Mathf.Clamp(Axis("Mouse Y"), -1f, 1f);

            // Key fallbacks if axes are not configured.
            if (Mathf.Abs(s.moveX) < 0.01f)
            {
                if (Key("A") || Key("LeftArrow")) s.moveX = -1f;
                if (Key("D") || Key("RightArrow")) s.moveX = 1f;
            }
            if (Mathf.Abs(s.moveY) < 0.01f)
            {
                if (Key("S") || Key("DownArrow")) s.moveY = -1f;
                if (Key("W") || Key("UpArrow")) s.moveY = 1f;
            }

            s.fire = MouseButton(0);
            s.ability = MouseButton(1);
            s.interact = Key("E");

            bool clickDown = MouseButtonDown(0);
            if (clickDown)
            {
                try
                {
                    var mp = MousePos(); // bottom-left origin
                    float sw = Mathf.Max(1f, (float)Screen.width);
                    float sh = Mathf.Max(1f, (float)Screen.height);
                    s.uiClick = true;
                    s.clickNX = Mathf.Clamp01(mp.x / sw);
                    s.clickNY = Mathf.Clamp01(1f - (mp.y / sh)); // top-left origin
                }
                catch { }
            }
            return s;
        }
    }
}
