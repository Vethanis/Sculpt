using System.Collections.Generic;
using Unity.Collections;
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
    public float3           m_center;
    public float3           m_size;
    public DistanceFunction m_function;
    public DistanceBlend    m_blend;
    public float            m_smoothness;

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
    
    public readonly float3     m_origin;
    public readonly float      m_radius;
    readonly int        m_dimension;
    readonly float      m_pitch;
    readonly float      m_pointSize;
    NativeArray<uint>   m_ran;
    NativeArray<Vert>   m_positions;
    NativeArray<Vert>   m_normals;

    [ReadOnly]
    NativeArray<CSG>    m_list;

    public CSGJob(
        float3      center, 
        float       radius, 
        float       ptSize, 
        int         dimension,
        List<CSG>   items)
    {
        int capacity = dimension * dimension * dimension;

        m_origin = center - radius;
        m_radius = radius;
        m_pointSize = ptSize;

        m_dimension = dimension;
        m_pitch = (m_radius * 2.0f) / dimension;

        m_list = new NativeArray<CSG>(
            items.ToArray(), 
            Allocator.TempJob);
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
    }
    public JobHandle Start()
    {
        return this.Schedule(m_dimension * m_dimension * m_dimension, m_dimension);
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
    public Mesh CreateMesh(JobHandle handle)
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

        var mesh = new Mesh();

        mesh.vertices = verts;
        mesh.normals = norms;
        mesh.triangles = inds;

        return mesh;
    }
};

public class Cell
{
    const float PointSize = 0.1f;
    const int   Dimension = 20;

    public static Material      ms_material;
    public static List<Cell>    ms_cells = new List<Cell>();

    List<CSG>   m_csgs;
    Mesh        m_mesh;
    CSGJob      m_job;
    JobHandle   m_handle;
    bool        m_running;
    bool        m_hasInserted;

    public Cell()
    {
        m_csgs = new List<CSG>();
        m_running = false;
        m_hasInserted = false;
    }
    public void Start(float3 center, float radius)
    {
        m_job = new CSGJob(
            center,
            radius,
            PointSize,
            Dimension,
            m_csgs);

        m_handle = m_job.Start();
        m_running = true;
    }
    public void Add(CSG csg)
    {
        m_csgs.Add(csg);
    }
    public bool IsCompleted()
    {
        return m_handle.IsCompleted;
    }
    public bool IsRunning()
    {
        return m_running;
    }
    public void Finish()
    {
        if(m_running)
        {
            m_mesh = m_job.CreateMesh(m_handle);
            m_running = false;

            if (!m_hasInserted)
            {
                ms_cells.Add(this);
                m_hasInserted = true;
            }
        }
    }
    public float3 GetPosition()
    {
        return m_job.m_origin + m_job.m_radius;
    }
    public static void Draw()
    {
        // Camera.current flickers :(
        var cam = GameObject.Find("Main Camera");
        var xform = cam.transform;
        var fwd = xform.forward;
        var eye = xform.position;
        foreach(var cell in ms_cells)
        {
            if(cell.m_mesh.vertices.Length == 0)
            {
                continue;
            }

            var V = normalize(cell.GetPosition() - (float3)eye);
            float d = dot(fwd, V);
            if(d > 0.0f)
            {
                Graphics.DrawMesh(cell.m_mesh, Matrix4x4.identity, ms_material, 0);
            }
        }
    }
    public static void ClearAll()
    {
        foreach(var cell in ms_cells)
        {
            cell.Finish();
        }
        ms_cells.Clear();
    }
};

public class OctNode
{
    const int MaxDepth = 6;

    static Queue<OctNode>   ms_dirtyQueue = new Queue<OctNode>();
    static Queue<Cell>      ms_workingQueue = new Queue<Cell>();

    OctNode[]       m_children;
    Cell            m_cell;
    readonly float3 m_center;
    readonly float  m_radius;
    readonly int    m_depth;

    public OctNode(float3 center, float radius, int depth = 0)
    {
        m_children = null;
        m_cell = null;
        m_center = center;
        m_radius = radius;
        m_depth = depth;

        if(m_depth == MaxDepth)
        {
            m_cell = new Cell();
        }
    }
    void EnsureChildren()
    {
        if (m_children == null)
        {
            m_children = new OctNode[8];
            float radius = m_radius * 0.5f;
            for (int i = 0; i < 8; ++i)
            {
                float3 center = m_center;
                center.x += (i & 1) == 0 ? -radius : radius;
                center.y += (i & 2) == 0 ? -radius : radius;
                center.z += (i & 4) == 0 ? -radius : radius;
                m_children[i] = new OctNode(center, radius, m_depth + 1);
            }
        }
    }
    bool Contains(ref CSG csg)
    {
        return csg.Distance(m_center) < m_radius * Util.CornerConstant;
    }
    public void Insert(CSG csg)
    {
        if(!Contains(ref csg))
        {
            return;
        }

        if(m_depth == MaxDepth)
        {
            m_cell.Add(csg);
            ms_dirtyQueue.Enqueue(this);
            return;
        }

        EnsureChildren();

        for(int i = 0; i < 8; ++i)
        {
            m_children[i].Insert(csg);
        }
    }
    public static void UpdateJobs()
    {
        int dirtyCount = ms_dirtyQueue.Count;
        int workCount = ms_workingQueue.Count;

        for (int i = 0; i < dirtyCount; ++i)
        {
            var node = ms_dirtyQueue.Dequeue();
            var cell = node.m_cell;
            if(cell.IsCompleted() & !cell.IsRunning())
            {
                cell.Start(node.m_center, node.m_radius);
                ms_workingQueue.Enqueue(cell);
                continue;
            }
            ms_dirtyQueue.Enqueue(node);
        }

        JobHandle.ScheduleBatchedJobs();

        for (int i = 0; i < workCount; ++i)
        {
            var cell = ms_workingQueue.Dequeue();
            if(cell.IsCompleted())
            {
                cell.Finish();
            }
            else
            {
                ms_workingQueue.Enqueue(cell);
            }
        }
    }
    public static void FinishAll()
    {
        while(ms_dirtyQueue.Count > 0 || ms_workingQueue.Count > 0)
        {
            UpdateJobs();
        }
    }
};

public class Mesher : MonoBehaviour
{
    const float Radius      = 100.0f;
    const float GrowRate    = 1.05f;
    const float ShrinkRate  = 0.95f;

    public Camera       m_camera;
    public GameObject   m_brush;
    public Material     m_material;
    public Mesh         m_sphereShape;
    public Mesh         m_boxShape;

    OctNode             m_root;
    float3              m_center;
    float3              m_size          = 1.0f;
    float               m_armLength     = 10.0f;
    float               m_smoothness    = 0.5f;
    DistanceFunction    m_function      = DistanceFunction.Sphere;

    void Start()
    {
        m_center = GetComponent<Transform>().position;
        m_root = new OctNode(m_center, Radius);
        Cell.ms_material = m_material;
    }
    void Update()
    {
        ControlsUpdate();
        SetBrush();
        OctNode.UpdateJobs();
        Cell.Draw();
    }
    void OnDisable()
    {
        OctNode.FinishAll();
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
            ClearAll();
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
        m_root.Insert(csg);
    }
    void ClearAll()
    {
        OctNode.FinishAll();
        Cell.ClearAll();
        m_root = new OctNode(m_center, Radius);
    }
};
