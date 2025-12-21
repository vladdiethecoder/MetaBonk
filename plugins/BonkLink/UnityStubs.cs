// UnityStubs.cs - Minimal Unity type stubs for BepInEx IL2CPP plugin compilation
// These stubs are replaced at runtime by the actual IL2CPP interop types

namespace UnityEngine
{
    public enum HideFlags
    {
        None = 0,
        HideAndDontSave = 61,
    }

    public struct Vector3
    {
        public float x, y, z;
        public static Vector3 zero => new Vector3();
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
    }

    public struct Rect
    {
        public float x, y, width, height;
        public Rect(float x, float y, float w, float h) { this.x = x; this.y = y; width = w; height = h; }
    }

    public class Object
    {
        public static void Destroy(Object obj) { }
        public static void DontDestroyOnLoad(Object target) { }
    }

    public class Component : Object
    {
        public T GetComponent<T>() where T : Component => default;
    }

    public class Behaviour : Component { }

    public class MonoBehaviour : Behaviour
    {
        public MonoBehaviour() { }
        public MonoBehaviour(System.IntPtr ptr) { }
        protected static T FindObjectOfType<T>() where T : Object => default;
    }

    public class GameObject : Object
    {
        public HideFlags hideFlags { get; set; }
        public GameObject(string name) { }
        public T AddComponent<T>() where T : Component => default;
    }

    public class Texture : Object { }

    public class Texture2D : Texture
    {
        public int width => 0;
        public int height => 0;
        public Texture2D(int width, int height, TextureFormat format, bool mipChain) { }
        public void ReadPixels(Rect source, int destX, int destY) { }
        public void ReadPixels(Rect source, int destX, int destY, bool recalculateMipMaps) { }
        public void Apply() { }
        public void Apply(bool updateMipmaps, bool makeNoLongerReadable) { }
        public byte[] EncodeToJPG(int quality) => System.Array.Empty<byte>();
    }

    public enum TextureFormat { RGB24 = 3 }

    public enum RenderTextureFormat
    {
        ARGB32 = 0,
    }

    public class RenderTexture : Texture
    {
        public int width => 0;
        public int height => 0;
        public RenderTexture(int width, int height, int depth, RenderTextureFormat format) { }
        public void Create() { }
        public void Release() { }
        public static RenderTexture active { get; set; }
    }

    public static class Graphics
    {
        public static void Blit(Texture source, RenderTexture dest) { }
    }

    public static class ScreenCapture
    {
        public static Texture2D CaptureScreenshotAsTexture() => null;
    }

    public static class Screen
    {
        public static int width => 1920;
        public static int height => 1080;
    }

    public static class Mathf
    {
        public static int Max(int a, int b) => a > b ? a : b;
    }

    public static class Time
    {
        public static float time => 0f;
        public static float unscaledTime => 0f;
    }
}
