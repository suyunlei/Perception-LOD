using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor; // 用于 Unity 编辑器中触发退出逻辑

public class DataExporter : MonoBehaviour
{
    public EyeTrackingRecorder eyeTrackingRecorder; // 引用 EyeTrackingRecorder
    private List<string> data = new List<string>();

    void Start()
    {
        // 添加 CSV 文件的标题行
        data.Add("Time,WorldPosX,WorldPosY,WorldPosZ,UVCoordX,UVCoordY,HitObjectName");
        
        // 定时记录眼动数据
        InvokeRepeating("RecordEyeData", 0f, 15f);
    }

    void RecordEyeData()
    {
        // 确保 EyeTrackingRecorder 正确绑定
        if (eyeTrackingRecorder == null)
        {
            Debug.LogError("EyeTrackingRecorder 未绑定到 DataExporter。");
            return;
        }

        // 获取数据
        Vector3 worldPos = eyeTrackingRecorder.eyeWorldPosition;
        Vector2 uvCoords = eyeTrackingRecorder.eyeUVCoordinates;
        string hitObjectName = eyeTrackingRecorder.hitObject != null ? eyeTrackingRecorder.hitObject.name : "No Hit";

        // 格式化记录
        string record = $"{Time.time},{worldPos.x},{worldPos.y},{worldPos.z},{uvCoords.x},{uvCoords.y},{hitObjectName}";
        data.Add(record);
        Debug.Log(record);
    }

    void ExportData()
    {
        // 检查数据列表是否为空
        if (data == null || data.Count == 0)
        {
            Debug.LogWarning("数据列表为空，未导出任何数据。");
            return;
        }

        // 确定文件路径
        string path = Path.Combine(Application.persistentDataPath, "EyeTrackingData.csv");
        Debug.Log($"文件路径: {path}");

        try
        {
            // 写入文件
            File.WriteAllLines(path, data.ToArray());
            Debug.Log($"数据成功导出到 {path}");

            // 检查文件是否存在
            if (File.Exists(path))
            {
                Debug.Log("文件已成功创建。");
            }
            else
            {
                Debug.LogError("文件未创建，请检查写入权限或路径。");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"数据导出失败: {ex.Message}");
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("应用程序退出，正在导出数据...");
        ExportData();
    }

#if UNITY_EDITOR
    // 用于编辑器中自动退出时的导出逻辑
    void OnDisable()
    {
        if (!EditorApplication.isPlayingOrWillChangePlaymode)
        {
            Debug.Log("编辑器退出，正在导出数据...");
            ExportData();
        }
    }
#endif
}