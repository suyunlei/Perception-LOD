using UnityEngine;
using Unity.XR.PXR;
using UnityEngine.XR;

public class EyeTrackingRecorder : MonoBehaviour
{
    public LineRenderer lineRenderer; // 引用 LineRenderer 组件
    public Transform xrOrigin; // XR Origin 位置
    public GameObject[] spheres; // 存储全景球体
    public GameObject spotlight; // Spotlight 物体

    public Vector3 eyeWorldPosition; // 眼动世界坐标
    public Vector2 eyeUVCoordinates; // 眼动UV坐标

    public GameObject hitObject; // 保存命中的物体

    private Vector3 combineEyeGazeVector;
    private Vector3 combineEyeGazeOrigin;
    private Matrix4x4 headPoseMatrix;
    private Matrix4x4 originPoseMatrix;

    private Vector3 combineEyeGazeOriginInWorldSpace;
    private Vector3 combineEyeGazeVectorInWorldSpace;

    void Start()
    {
        // 如果 LineRenderer 未绑定，尝试获取组件
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.GetComponent<LineRenderer>();
            if (lineRenderer == null)
            {
                lineRenderer = gameObject.AddComponent<LineRenderer>();
            }
        }
        // 初始化 LineRenderer 属性
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default")) { color = Color.green };
    }

    void Update()
    {
        // 确保 xrOrigin 已正确设置
        if (xrOrigin == null)
        {
            Debug.LogError("xrOrigin 未设置，请检查 EyeTrackingRecorder 的 xrOrigin 引用！");
            return;
        }

        // 更新 LineRenderer 的位置
        if (lineRenderer != null)
        {
            lineRenderer.SetPosition(0, eyeWorldPosition); // 射线起点
            lineRenderer.SetPosition(1, eyeWorldPosition + combineEyeGazeVectorInWorldSpace * 100.0f); // 射线终点
        }

        // 更新 XR Origin 的世界变换矩阵
        originPoseMatrix = xrOrigin.localToWorldMatrix;

        // 获取眼动数据
        if (!PXR_EyeTracking.GetHeadPosMatrix(out headPoseMatrix))
        {
            Debug.LogWarning("获取头部位置矩阵失败。");
            return;
        }

        if (!PXR_EyeTracking.GetCombineEyeGazeVector(out combineEyeGazeVector) ||
            !PXR_EyeTracking.GetCombineEyeGazePoint(out combineEyeGazeOrigin))
        {
            Debug.LogWarning("获取眼动追踪数据失败。");
            return;
        }

        // 打印眼动数据,乘以1000获取原始值。api里面自己除以了1000
        Debug.Log($"combineEyeGazeVector: {combineEyeGazeVector*1000}, combineEyeGazeOrigin: {combineEyeGazeOrigin*1000}");

        // 转换为世界坐标
        combineEyeGazeOriginInWorldSpace = originPoseMatrix.MultiplyPoint(headPoseMatrix.MultiplyPoint(combineEyeGazeOrigin));
        combineEyeGazeVectorInWorldSpace = originPoseMatrix.MultiplyVector(headPoseMatrix.MultiplyVector(combineEyeGazeVector));

        // 更新眼动的世界坐标
        eyeWorldPosition = combineEyeGazeOriginInWorldSpace;

        // 调试打印世界坐标
        Debug.Log($"眼动原点世界坐标: {eyeWorldPosition}");

        // 更新 Spotlight 的位置和方向
        if (spotlight != null)
        {
            spotlight.transform.position = eyeWorldPosition;
            spotlight.transform.rotation = Quaternion.LookRotation(combineEyeGazeVectorInWorldSpace, Vector3.up);
        }

        // 使用 Raycast 进行检测（确保检测球体内部）
        Ray ray = new Ray(eyeWorldPosition, combineEyeGazeVectorInWorldSpace);
        float maxDistance = 100.0f; // 增大检测范围
        RaycastHit hit;

        Debug.Log($"射线起点: {eyeWorldPosition}, 射线方向: {combineEyeGazeVectorInWorldSpace}");
        if (Physics.Raycast(ray, out hit, maxDistance))
        {
            // 保存命中的物体
            hitObject = hit.collider.gameObject;
            Debug.Log($"射线命中物体: {hitObject.name}");

            MeshCollider meshCollider = hit.collider as MeshCollider;
            if (meshCollider != null && meshCollider.sharedMesh != null)
            {
                // 获取 UV 坐标
                Vector2 uvCoordinates = hit.textureCoord;
                if (uvCoordinates != Vector2.zero)
                {
                    eyeUVCoordinates = uvCoordinates;
                    Debug.Log($"UV 坐标: {eyeUVCoordinates}");
                }
                else
                {
                    Debug.LogWarning("UV 坐标为零，请检查模型的 UV 映射。");
                }
            }
            else
            {
                Debug.LogWarning("命中物体不是 MeshCollider 或未设置共享网格。");
            }

            // 打印命中的世界坐标
            Debug.Log($"命中点的世界坐标: {hit.point}");
        }
        else
        {
            // 未命中任何对象
            hitObject = null;
            Debug.LogWarning("射线未击中任何对象。");
        }

        // 可视化调试射线
        Debug.DrawRay(eyeWorldPosition, combineEyeGazeVectorInWorldSpace * 100.0f, Color.green);
    }
}