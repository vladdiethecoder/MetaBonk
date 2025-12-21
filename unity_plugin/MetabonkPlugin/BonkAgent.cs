using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Metabonk;

/// <summary>
/// ML-Agents agent that controls the MegaBonk player.
/// In recovery mode, game-specific lookups are performed via reflection.
/// Update the reflection helpers after inspecting BepInEx/interop proxies.
/// </summary>
public sealed class BonkAgent : Agent
{
    private const int Rays = 30;
    private const float RayDistance = 25f;
    private readonly float[] _rayObs = new float[Rays * 2];

    private Rigidbody? _rb;

    // Reward weights (curriculum-controlled)
    private float _survivalTick = 0.001f;
    private float _velocityWeight = 0.0001f;
    private float _damagePenalty = -1f;

    public override void Initialize()
    {
        _rb = GetComponent<Rigidbody>();
        LoadCurriculumFromEnv();
        base.Initialize();
    }

    public override void OnEpisodeBegin()
    {
        LoadCurriculumFromEnv();
        base.OnEpisodeBegin();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Proprioception
        sensor.AddObservation(GetHealthNorm());
        sensor.AddObservation(GetShieldNorm());
        sensor.AddObservation(GetMaxHealthNorm());

        // Kinematics
        var vel = _rb != null ? _rb.velocity : Vector3.zero;
        sensor.AddObservation(vel.x);
        sensor.AddObservation(vel.y);
        sensor.AddObservation(vel.z);
        sensor.AddObservation(IsGrounded() ? 1f : 0f);
        sensor.AddObservation(transform.eulerAngles.y / 360f);

        // Resources / economy (fallback to 0)
        sensor.AddObservation(GetXpNorm());
        sensor.AddObservation(GetLevelNorm());
        sensor.AddObservation(GetGoldNorm());
        sensor.AddObservation(GetSilverNorm());

        // Objectives
        var bossVec = GetVectorToBoss();
        sensor.AddObservation(bossVec.x);
        sensor.AddObservation(bossVec.y);
        sensor.AddObservation(bossVec.z);

        var shrineVec = GetVectorToNearestShrine();
        sensor.AddObservation(shrineVec.x);
        sensor.AddObservation(shrineVec.y);
        sensor.AddObservation(shrineVec.z);

        // Cooldowns
        sensor.AddObservation(GetDashCooldownNorm());
        sensor.AddObservation(GetUltChargeNorm());

        // Lidar-style rays (distance, tag id)
        CastRays();
        for (int i = 0; i < _rayObs.Length; i++)
            sensor.AddObservation(_rayObs[i]);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var cont = actions.ContinuousActions;
        var disc = actions.DiscreteActions;

        float moveX = Mathf.Clamp(cont[0], -1f, 1f);
        float moveY = Mathf.Clamp(cont[1], -1f, 1f);

        bool jump = disc[0] == 1;
        bool dash = disc[1] == 1;
        bool interact = disc[2] == 1;

        ApplyMovement(moveX, moveY);
        if (jump) ApplyJump();
        if (dash) ApplyDash();
        if (interact) ApplyInteract();

        // Reward shaping
        AddReward(_survivalTick); // existential survival
        AddReward(_velocityWeight * velMag());

        if (TookDamageThisTick())
            AddReward(_damagePenalty);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Intentionally avoid hard-coded human control mappings (WASD/Space/etc.).
        // If you want manual control, implement a separate input adapter.
        var cont = actionsOut.ContinuousActions;
        for (int i = 0; i < cont.Length; i++)
            cont[i] = 0f;
        var disc = actionsOut.DiscreteActions;
        for (int i = 0; i < disc.Length; i++)
            disc[i] = 0;
    }

    // --- Ray perception ---

    private void CastRays()
    {
        var origin = transform.position + Vector3.up * 0.5f;
        for (int i = 0; i < Rays; i++)
        {
            float ang = i * (360f / Rays);
            var dir = Quaternion.Euler(0, ang, 0) * transform.forward;
            if (Physics.Raycast(origin, dir, out RaycastHit hit, RayDistance))
            {
                _rayObs[i * 2] = hit.distance / RayDistance;
                _rayObs[i * 2 + 1] = TagToId(hit.collider.tag);
            }
            else
            {
                _rayObs[i * 2] = 1f;
                _rayObs[i * 2 + 1] = 0f;
            }
        }
    }

    private static float TagToId(string tag) =>
        tag switch
        {
            "Enemy" => 1f,
            "Wall" => 2f,
            "XP" => 3f,
            "Boss" => 4f,
            _ => 0f
        };

    // --- Game-specific reflection helpers (TODO update) ---

    private float GetHealthNorm() => TryReadFloat("health", 1f);
    private float GetMaxHealthNorm() => TryReadFloat("maxHealth", 1f);
    private float GetShieldNorm() => TryReadFloat("shield", 0f);

    private float GetXpNorm() => TryReadFloat("xp", 0f);
    private float GetLevelNorm() => TryReadFloat("level", 0f);
    private float GetGoldNorm() => TryReadFloat("gold", 0f);
    private float GetSilverNorm() => TryReadFloat("silver", 0f);

    private float GetDashCooldownNorm() => TryReadFloat("dashCooldown", 0f);
    private float GetUltChargeNorm() => TryReadFloat("ultCharge", 0f);

    private bool IsGrounded() => TryReadBool("isGrounded", true);

    private Vector3 GetVectorToBoss() => Vector3.zero;
    private Vector3 GetVectorToNearestShrine() => Vector3.zero;

    private void ApplyMovement(float x, float y)
    {
        // TODO: Write to input buffer / PlayerInput struct.
        var move = new Vector3(x, 0, y);
        transform.position += transform.TransformDirection(move) * Time.fixedDeltaTime * 8f;
    }

    private void ApplyJump()
    {
        if (_rb != null && IsGrounded())
            _rb.AddForce(Vector3.up * 6f, ForceMode.VelocityChange);
    }

    private void ApplyDash() { }
    private void ApplyInteract() { }

    private bool TookDamageThisTick() => false;

    private float velMag() => _rb != null ? _rb.velocity.magnitude : 0f;

    private float TryReadFloat(string fieldName, float fallback)
    {
        try
        {
            var comp = GetComponents<MonoBehaviour>().FirstOrDefault();
            if (comp == null) return fallback;
            var f = comp.GetType().GetField(fieldName);
            if (f != null && f.GetValue(comp) is float v) return v;
        }
        catch { }
        return fallback;
    }

    private bool TryReadBool(string fieldName, bool fallback)
    {
        try
        {
            var comp = GetComponents<MonoBehaviour>().FirstOrDefault();
            if (comp == null) return fallback;
            var f = comp.GetType().GetField(fieldName);
            if (f != null && f.GetValue(comp) is bool v) return v;
        }
        catch { }
        return fallback;
    }

    private void LoadCurriculumFromEnv()
    {
        _survivalTick = ReadEnvFloat("METABONK_SURVIVAL_TICK", _survivalTick);
        _velocityWeight = ReadEnvFloat("METABONK_VELOCITY_WEIGHT", _velocityWeight);
        _damagePenalty = ReadEnvFloat("METABONK_DAMAGE_PENALTY", _damagePenalty);
    }

    private static float ReadEnvFloat(string key, float fallback)
    {
        var s = Environment.GetEnvironmentVariable(key);
        if (string.IsNullOrWhiteSpace(s))
            return fallback;
        return float.TryParse(s, out var v) ? v : fallback;
    }
}
