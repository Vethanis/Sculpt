using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Burst;
using Unity.Rendering;
using UnityEngine.Rendering;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Jobs;

class Util
{
    public static float3 Tangent(float3 N)
    {
        if (abs(N.x) > 0.001f)
        {
            return normalize(cross(new float3(0.0f, 1.0f, 0.0f), N));
        }
        return normalize(cross(new float3(1.0f, 0.0f, 0.0f), N));
    }
    public static float3 Bitangent(float3 N, float3 T)
    {
        return normalize(cross(N, T));
    }
    // AABB vs AABB intersection test
    public static bool Intersects(float3 loA, float3 hiA, float3 loB, float3 hiB)
    {
        bool3 a = (loB < hiA) & (hiB > loA);
        return a.x & a.y & a.z;
    }
    // distance from center of cube to corner, aka sqrt(3)
    public const float CornerConstant = 1.732051f;
};

public enum DistanceFunction
{
    Sphere = 0,
    Box,
};

public enum DistanceBlend
{
    Add = 0,
    Sub,
    SmoothAdd,
    SmoothSub,
}

// Represents a signed distance function instantiation
// with position, size, shape, blending, and smoothness
public struct CSG
{
    public static NativeList<CSG> ms_pool = new NativeList<CSG>(Allocator.Persistent);

    public float3           m_center;
    public float3           m_size;
    public DistanceFunction m_function;
    public DistanceBlend    m_blend;
    public float            m_smoothness;

    public static int Create(CSG csg)
    {
        int id = ms_pool.Length;
        ms_pool.Add(csg);
        return id;
    }
    public float3 Min()
    {
        return m_center - m_size;
    }
    public float3 Max()
    {
        return m_center + m_size;
    }
    public float Distance(float3 pt)
    {
        switch (m_function)
        {
            default:
            case DistanceFunction.Sphere:
            {
                return DistanceSphere(pt, m_center, m_size.x);
            }
            case DistanceFunction.Box:
            {
                return DistanceBox(pt, m_center, m_size);
            }
        };
    }
    public float Blend(float a, float b)
    {
        switch (m_blend)
        {
            default:
            case DistanceBlend.Add:
            {
                return BlendAdd(a, b);
            }
            case DistanceBlend.Sub:
            {
                return BlendSub(a, b);
            }
            case DistanceBlend.SmoothAdd:
            {
                return BlendSmoothAdd(a, b, m_smoothness);
            }
            case DistanceBlend.SmoothSub:
            {
                return BlendSmoothSub(a, b, m_smoothness);
            }
        }
    }
    public float3 Normal(float3 pt)
    {
        const float e = 0.001f;
        float3 v = new float3(e, 0.0f, 0.0f);
        float dx = Distance(pt + v) - Distance(pt - v);
        v.x = 0.0f;
        v.y = e;
        float dy = Distance(pt + v) - Distance(pt - v);
        v.y = 0.0f;
        v.z = e;
        float dz = Distance(pt + v) - Distance(pt - v);
        v.x = dx;
        v.y = dy;
        v.z = dz;
        return normalize(v);
    }
    public static float DistanceSphere(float3 pt, float3 center, float radius)
    {
        return distance(pt, center) - radius;
    }
    public static float DistanceBox(float3 pt, float3 center, float3 dim)
    {
        float3 d = abs(pt - center) - dim;
        return min(cmax(d), 0.0f) + length(max(d, 0.0f));
    }
    public static float BlendAdd(float a, float b)
    {
        return min(a, b);
    }
    public static float BlendSub(float a, float b)
    {
        return max(a, b);
    }
    public static float BlendSmoothAdd(float a, float b, float smoothness)
    {
        float e = max(smoothness - abs(a - b), 0.0f);
        return min(a, b) - e * e * 0.25f / smoothness;
    }
    public static float BlendSmoothSub(float a, float b, float smoothness)
    {
        float c = BlendSmoothAdd(-a, b, smoothness);
        return -c;
    }
    public static void Shutdown()
    {
        ms_pool.Dispose();
    }
};

[BurstCompile]
public struct CSGJob : IJobParallelFor
{
    struct Vert
    {
        public float3 a;
        public float3 b;
        public float3 c;
        public float3 d;
    };
    
    public float3       m_origin;
    public float        m_radius;
    readonly int        m_dimension;
    float               m_pitch;
    float               m_pointSize;
    NativeArray<uint>   m_ran;
    NativeArray<Vert>   m_positions;
    NativeArray<Vert>   m_normals;

    [ReadOnly]
    NativeArray<CSG>    m_list;
    
    public CSGJob(
        float                   ptSize, 
        int                     dimension)
    {
        m_pointSize = ptSize;
        m_dimension = dimension;
        m_list      = default(NativeArray<CSG>);
        m_ran       = default(NativeArray<uint>);
        m_positions = default(NativeArray<Vert>);
        m_normals   = default(NativeArray<Vert>);
        m_origin    = default(float3);
        m_radius    = default(float);
        m_pitch     = default(float);
    }
    public JobHandle Start(
        float3          center, 
        float           radius,
        NativeList<int> list)
    {
        m_origin = center - radius;
        m_radius = radius;
        m_pitch = (m_radius * 2.0f) / m_dimension;

        int capacity = m_dimension * m_dimension * m_dimension;

        m_list = new NativeArray<CSG>(
            list.Length, 
            Allocator.TempJob,
            NativeArrayOptions.UninitializedMemory);
        for(int i = 0; i < list.Length; ++i)
        {
            m_list[i] = CSG.ms_pool[list[i]];
        }
        m_ran = new NativeArray<uint>(
            capacity,
            Allocator.TempJob,
            NativeArrayOptions.ClearMemory);
        m_positions = new NativeArray<Vert>(
            capacity,
            Allocator.TempJob,
            NativeArrayOptions.UninitializedMemory);
        m_normals = new NativeArray<Vert>(
            capacity,
            Allocator.TempJob,
            NativeArrayOptions.UninitializedMemory);

        return this.Schedule(capacity, m_dimension);
    }
    public float Distance(float3 pt)
    {
        float a = float.MaxValue;
        for(int i = 0; i < m_list.Length; ++i)
        {
            float b = m_list[i].Distance(pt);
            a = m_list[i].Blend(a, b);
        }
        return a;
    }
    public float3 Normal(float3 pt)
    {
        const float e = 0.001f;
        float3 v = new float3(e, 0.0f, 0.0f);
        float dx = Distance(pt + v) - Distance(pt - v);
        v.x = 0.0f;
        v.y = e;
        float dy = Distance(pt + v) - Distance(pt - v);
        v.y = 0.0f;
        v.z = e;
        float dz = Distance(pt + v) - Distance(pt - v);
        v.x = dx;
        v.y = dy;
        v.z = dz;
        return normalize(v);
    }
    public void Execute(int idx)
    {
        int dim = m_dimension;
        float pitch = m_pitch;
        float ptSize = m_pointSize;
        float radius = pitch * 0.5f * Util.CornerConstant;

        int x = idx % dim;
        int y = (idx / dim) % dim;
        int z = (idx / (dim * dim)) % dim;
        float3 P = m_origin + float3(x, y, z) * pitch;

        float distance = Distance(P);
        if (abs(distance) >= radius)
        {
            return;
        }

        m_ran[idx] = 1;

        float3 N = Normal(P);
        P = P - N * distance;
        float3 T = Util.Tangent(N);
        float3 B = Util.Bitangent(N, T);

        T *= ptSize;
        B *= ptSize;

        m_positions[idx] = new Vert
        {
            a = P - T - B,
            b = P - T + B,
            c = P + T + B,
            d = P + T - B,
        };

        m_normals[idx] = new Vert
        {
            a = Normal(m_positions[idx].a),
            b = Normal(m_positions[idx].b),
            c = Normal(m_positions[idx].c),
            d = Normal(m_positions[idx].d),
        };
    }
    public void Finish(JobHandle handle, Mesh mesh)
    {
        handle.Complete();

        int count = 0;
        foreach (var ran in m_ran)
        {
            ++count;
        }
        var verts = new Vector3[count * 4];
        var norms = new Vector3[count * 4];
        var inds = new int[count * 6];

        int j = 0;
        for (int i = 0; i < m_ran.Length; ++i)
        {
            if (m_ran[i] != 0)
            {
                var vert = m_positions[i];
                verts[j * 4 + 0] = vert.a;
                verts[j * 4 + 1] = vert.b;
                verts[j * 4 + 2] = vert.c;
                verts[j * 4 + 3] = vert.d;

                var norm = m_normals[i];
                norms[j * 4 + 0] = norm.a;
                norms[j * 4 + 1] = norm.b;
                norms[j * 4 + 2] = norm.c;
                norms[j * 4 + 3] = norm.d;

                inds[j * 6 + 0] = j * 4 + 0;
                inds[j * 6 + 1] = j * 4 + 2;
                inds[j * 6 + 2] = j * 4 + 1;

                inds[j * 6 + 3] = j * 4 + 0;
                inds[j * 6 + 4] = j * 4 + 3;
                inds[j * 6 + 5] = j * 4 + 2;

                ++j;
            }
        }

        m_list.Dispose();
        m_ran.Dispose();
        m_positions.Dispose();
        m_normals.Dispose();
        
        mesh.vertices   = verts;
        mesh.normals    = norms;
        mesh.triangles  = inds;

        mesh.RecalculateBounds();
    }
};

public unsafe struct Leaf
{
    const float PointSize = 0.15f;
    const int   Dimension = 8;

    public static Material              ms_material;
    static NativeList<JobHandle>        ms_handles      = new NativeList<JobHandle>(Allocator.Persistent);
    static NativeList<float3>           ms_positions    = new NativeList<float3>(Allocator.Persistent);
    static List<Mesh>                   ms_meshes       = new List<Mesh>();
    static List<CSGJob>                 ms_jobs         = new List<CSGJob>();
    static List<NativeList<int>>        ms_items        = new List<NativeList<int>>();
    
    int             m_slot;
    int             m_running;

    public void Init()
    {
        var job = new CSGJob(
            PointSize,
            Dimension);

        m_slot = ms_handles.Length;
        ms_handles.Add(new JobHandle());
        ms_positions.Add(new float3());
        ms_meshes.Add(new Mesh());
        ms_jobs.Add(job);
        ms_items.Add(new NativeList<int>(Allocator.Persistent));

        m_running = 0;
    }
    public bool Start(float3 center, float radius)
    {
        if(m_running != 0)
        {
            return false;
        }
        ms_positions[m_slot] = center;
        var job = ms_jobs[m_slot];
        ms_handles[m_slot] = job.Start(center, radius, ms_items[m_slot]);
        ms_jobs[m_slot] = job;
        m_running = 1;
        return true;
    }
    public void Add(int csg)
    {
        ms_items[m_slot].Add(csg);
    }
    public bool Finish()
    {
        if(m_running != 0)
        {
            if(ms_handles[m_slot].IsCompleted)
            {
                var job = ms_jobs[m_slot];
                job.Finish(ms_handles[m_slot], ms_meshes[m_slot]);
                ms_jobs[m_slot] = job;
                m_running = 0;
                return true;
            }
            return false;
        }
        return true;
    }
    public static void Draw()
    {
        // Camera.current flickers :(
        var cam = GameObject.Find("Main Camera");
        var fov = cam.GetComponent<Camera>().fieldOfView;
        fov = radians(fov);
        var xform = cam.transform;
        float3 fwd = xform.forward;
        float3 eye = xform.position;
        
        for(int i = 0; i < ms_meshes.Count; ++i)
        {
            var mesh = ms_meshes[i];
            if(mesh.vertices.Length > 0)
            {
                var pos = ms_positions[i];
                var V = normalize(pos - eye);
                float d = dot(fwd, V);
                if(acos(d) < fov)
                {
                    Graphics.DrawMesh(mesh, Matrix4x4.identity, ms_material, 0);
                }
            }
        }
    }
    public static void Shutdown()
    {
        for(int i = 0; i < ms_items.Count; ++i)
        {
            ms_items[i].Dispose();
        }
        ms_handles.Dispose();
        ms_positions.Dispose();
        ms_meshes.Clear();
        ms_jobs.Clear();
        ms_items.Clear();
        CSG.Shutdown();
    }
};

public unsafe struct OctNode
{
    const int MaxDepth = 6;

    public static NativeList<OctNode>   ms_pool             = new NativeList<OctNode>(Allocator.Persistent);
    static NativeQueue<int>             ms_dirtyNodes       = new NativeQueue<int>(Allocator.Persistent);
    static NativeQueue<int>             ms_workingCells     = new NativeQueue<int>(Allocator.Persistent);

    fixed int       m_children[8];
    Leaf            m_leaf;
    readonly float3 m_center;
    readonly float  m_radius;
    readonly int    m_depth;

    public static int Create(OctNode node)
    {
        int id = ms_pool.Length;
        ms_pool.Add(node);
        return id;
    }
    public OctNode(
        float3  center, 
        float   radius, 
        int     depth = 0)
    {
        fixed (int* c = m_children)
        {
            c[0] = -1;
        }
        m_center = center;
        m_radius = radius;
        m_depth = depth;

        m_leaf = default(Leaf);
        if (m_depth == MaxDepth)
        {
            m_leaf = new Leaf();
            m_leaf.Init();
        }
    }
    void EnsureChildren()
    {
        fixed(int* c = m_children)
        {
            if (c[0] == -1)
            {
                float radius = m_radius * 0.5f;
                for (int i = 0; i < 8; ++i)
                {
                    float3 center = m_center;
                    center.x += (i & 1) == 0 ? -radius : radius;
                    center.y += (i & 2) == 0 ? -radius : radius;
                    center.z += (i & 4) == 0 ? -radius : radius;
                    c[i] = OctNode.Create(new OctNode(center, radius, m_depth + 1));
                }
            }
        }
    }
    bool Contains(CSG csg)
    {
        return csg.Distance(m_center) < m_radius * Util.CornerConstant;
    }
    public void Insert(int nodeID, int csgID)
    {
        if(!Contains(CSG.ms_pool[csgID]))
        {
            return;
        }
        
        if(m_depth == MaxDepth)
        {
            m_leaf.Add(csgID);
            ms_dirtyNodes.Enqueue(nodeID);
            return;
        }

        EnsureChildren();

        fixed(int* c = m_children)
        {
            for (int i = 0; i < 8; ++i)
            {
                int id = c[i];
                var node = ms_pool[id];
                node.Insert(id, csgID);
                ms_pool[id] = node;
            }
        }
    }
    public static void UpdateJobs()
    {
        int dirtyCount = ms_dirtyNodes.Count;
        int workCount = ms_workingCells.Count;

        for (int i = 0; i < dirtyCount; ++i)
        {
            var nodeID = ms_dirtyNodes.Dequeue();
            var node = ms_pool[nodeID];

            bool started = node.m_leaf.Start(node.m_center, node.m_radius);

            ms_pool[nodeID] = node;

            if(started)
            {
                ms_workingCells.Enqueue(nodeID);
            }
            else
            {
                ms_dirtyNodes.Enqueue(nodeID);
            }
        }

        JobHandle.ScheduleBatchedJobs();

        for (int i = 0; i < workCount; ++i)
        {
            var nodeID = ms_workingCells.Dequeue();
            var node = ms_pool[nodeID];
            bool finished = node.m_leaf.Finish();
            ms_pool[nodeID] = node;
            if(!finished)
            {
                ms_workingCells.Enqueue(nodeID);
            }
        }
    }
    public static void FinishAll()
    {
        while(ms_dirtyNodes.Count > 0 || ms_workingCells.Count > 0)
        {
            UpdateJobs();
        }
    }
    public static void Shutdown()
    {
        for(int i = 0; i < ms_pool.Length; ++i)
        {
            ms_pool[i].m_leaf.Finish();
        }
        ms_pool.Dispose();
        ms_dirtyNodes.Dispose();
        ms_workingCells.Dispose();
        Leaf.Shutdown();
    }
};

public class Mesher : MonoBehaviour
{
    const float Radius      = 50.0f;
    const float GrowRate    = 1.05f;
    const float ShrinkRate  = 0.95f;

    public Camera       m_camera;
    public GameObject   m_brush;
    public Material     m_material;
    public Mesh         m_sphereShape;
    public Mesh         m_boxShape;

    int                 m_root;
    float3              m_center;
    float3              m_size          = 1.0f;
    float               m_armLength     = 10.0f;
    float               m_smoothness    = 0.5f;
    DistanceFunction    m_function      = DistanceFunction.Sphere;

    void Start()
    {
        m_center = GetComponent<Transform>().position;
        m_root = OctNode.Create(new OctNode(m_center, Radius));
        Leaf.ms_material = m_material;
    }
    void Update()
    {
        ControlsUpdate();
        SetBrush();
        OctNode.UpdateJobs();
        Leaf.Draw();
    }
    void OnDisable()
    {
        OctNode.Shutdown();
    }
    void ControlsUpdate()
    {
        if (Input.GetMouseButton(0))
        {
            AddCSG(true);
        }
        else if (Input.GetMouseButton(1))
        {
            AddCSG(false);
        }
        if(Input.GetKey(KeyCode.Z))
        {
            //ClearAll();
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            if (m_function == DistanceFunction.Sphere)
            {
                m_function = DistanceFunction.Box;
                m_brush.GetComponent<MeshFilter>().mesh = m_boxShape;
            }
            else
            {
                m_function = DistanceFunction.Sphere;
                m_brush.GetComponent<MeshFilter>().mesh = m_sphereShape;
            }
        }
        if (Input.GetKey(KeyCode.UpArrow))
        {
            m_size *= GrowRate;
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            m_size *= ShrinkRate;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            m_smoothness *= GrowRate;
        }
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            m_smoothness *= ShrinkRate;
        }
        if (Input.GetKey(KeyCode.U))
        {
            var color = m_material.color;
            color.r *= GrowRate;
            m_material.color = color;
        }
        if (Input.GetKey(KeyCode.J))
        {
            var color = m_material.color;
            color.r *= ShrinkRate;
            m_material.color = color;
        }
        if (Input.GetKey(KeyCode.I))
        {
            var color = m_material.color;
            color.g *= GrowRate;
            m_material.color = color;
        }
        if (Input.GetKey(KeyCode.K))
        {
            var color = m_material.color;
            color.g *= ShrinkRate;
            m_material.color = color;
        }
        if (Input.GetKey(KeyCode.O))
        {
            var color = m_material.color;
            color.b *= GrowRate;
            m_material.color = color;
        }
        if (Input.GetKey(KeyCode.L))
        {
            var color = m_material.color;
            color.b *= ShrinkRate;
            m_material.color = color;
        }
    }
    void SetBrush()
    {
        var xform = m_camera.gameObject.transform;
        float3 pos = xform.position + xform.forward * m_armLength;
        m_brush.transform.position = pos;
        m_brush.transform.localScale = m_size * 2.0f;
    }
    void AddCSG(bool additive)
    {
        var csg = new CSG
        {
            m_function = m_function,
            m_blend = additive ? DistanceBlend.SmoothAdd : DistanceBlend.SmoothSub,
            m_center = m_brush.transform.position,
            m_size = m_size,
            m_smoothness = m_smoothness,
        };
        var root = OctNode.ms_pool[m_root];
        root.Insert(m_root, CSG.Create(csg));
        OctNode.ms_pool[m_root] = root;
    }
};
