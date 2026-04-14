using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Unity.XR.PXR;
using UnityEngine.UI;
using System;
using TMPro; // 添加这个using，因为使用TMP_Text

public class ChangeImageAndSaveData : MonoBehaviour
{
    public Cubemap[] textures; // 现在扩展到39张
    public float waitTime; // 每张主要图片显示时间（原15s）
    public Cubemap positioningTexture; // 新增：定位图片 (zbx.jpg as Cubemap)
    public float positioningTime = 1f; // 新增：定位图片显示时间（默认1秒）
    public TMP_Text restText; // 新增：休息提示Text (TextMeshProUGUI)
    
    Vector3 hitPos;
    Vector2 hitUV;
    public Transform outStartPos;

    private bool isStartSave = false;
    private Material material;
    private int index;

    private bool support = false;
    private EyeTrackingMode[] eyeTrackingModes;

    private List<string[]> data = new List<string[]> // CSV
    {
        new string[] { "LOD","LeftPosition" ,"RightPosition" , "EyeDirection" , "Word(hitPosition)","hitUV" },
    };

    public LineRenderer lineRenderer;

    public int length;
    void Start()
    {
        material = RenderSettings.skybox;
        StartEyeTracking(true);
        ShuffleTextures(); // 随机打乱39张
        if (restText != null) restText.gameObject.SetActive(false); // 默认隐藏休息提示
    }

    public void StartGame(GameObject gameObject)
    {
        isStartSave = true;
        StartCoroutine(ChangeImage());
        gameObject.SetActive(false);
    }

    public void GameOver()
    {
        isStartSave = false;
        Application.Quit();
    }

    void StartEyeTracking(bool isStart)
    {
        if (isStart)
        {
            int supportModesCount = 0;
            PXR_MotionTracking.GetEyeTrackingSupported(ref support, ref supportModesCount, ref eyeTrackingModes);
            if (support)
            {
                EyeTrackingStartInfo eyeTrackingStartInfo = new EyeTrackingStartInfo();
                eyeTrackingStartInfo.needCalibration = 1;
                eyeTrackingStartInfo.mode = EyeTrackingMode.PXR_ETM_BOTH;
                PXR_MotionTracking.StartEyeTracking(ref eyeTrackingStartInfo);
            }
        }
        else
        {
            if (support)
            {
                EyeTrackingStopInfo eyeTrackingStopInfo = new EyeTrackingStopInfo();
                PXR_MotionTracking.StopEyeTracking(ref eyeTrackingStopInfo);
            }
        }
    }

    void RecordTrackData()
    {
        if (support)
        {
            long timMap = 0;
            Posef liftPos = new Posef();
            Posef rightPos = new Posef();
            PXR_MotionTracking.GetPerEyePose(ref timMap, ref liftPos, ref rightPos);

            Vector3 outGaze = Vector3.zero;
            Vector3 startPos = Vector3.zero;
            startPos = (liftPos.Position.ToVector3() + rightPos.Position.ToVector3()) / 2;

            PXR_EyeTracking.GetCombineEyeGazeVector(out outGaze);
            Vector3 stopPos = Camera.main.transform.rotation * outGaze;

            Ray ray = new Ray(startPos, stopPos * length);
            RaycastHit hit;
            float maxDistance = 1000f;
            
            if (Physics.Raycast(ray, out hit, maxDistance))
            {
                if (hit.collider == null)
                {
                    return;
                }
                else
                {
                    hitPos = hit.point;
                }
            }

            // test
            Vector3 startPos2 = hitPos * 10;
            Ray ray2 = new Ray(startPos2, hitPos - startPos2);
            RaycastHit hit2;
            float maxDistance2 = 1000f;

            if (Physics.Raycast(ray2, out hit2, maxDistance2))
            {
                if (hit2.collider == null)
                {
                    return;
                }
                else
                {
                    Vector2 uv = new Vector2(1 - hit2.textureCoord.x, hit2.textureCoord.y);
                    hitUV = uv;
                    // debugText.text = uv.ToString(); // 如果有debugText，保持原样
                }
            }
            Debug.DrawRay(transform.position, transform.forward * 1000, Color.blue);

            if (isStartSave)
            {
                string imageName = textures[index - 1].name; // 使用textures中的名字
                string liftPosition = liftPos.Position.ToVector3().ToString().Trim('(', ')');
                string rightPosition = rightPos.Position.ToVector3().ToString().Trim('(', ')');
                string eyeDirection = stopPos.ToString().Trim('(', ')');
                string hitPosition = hitPos.ToString().Trim('(', ')');
                string tetureCoord = string.Format("({0:F7}, {1:F7})", hitUV.x, hitUV.y);

                data.Add(new string[] { "\"" + imageName + "\"", "\"" + liftPosition + "\"", "\"" + rightPosition + "\"", "\"" + eyeDirection + "\"", "\"" + hitPosition + "\"", "\"" + tetureCoord + "\"" });
            }
        }
    }
    
    IEnumerator ChangeImage()
    {
        int pictureCount = 0; // 新增：计数器，跟踪已显示的主要图片数量
        index = 0; // 从0开始

        for (int i = 0; i < textures.Length; i++)
        {
            // 新增：如果不是第一张，在切换前显示定位图片（让眼睛复位）
            if (i > 0)
            {
                material.SetTexture("_Tex", positioningTexture); // 显示定位图片
                isStartSave = false; // 暂停数据记录
                yield return new WaitForSeconds(positioningTime); // 等待1秒（可调整）
                isStartSave = true; // 恢复数据记录
            }

            // 显示主要图片
            material.SetTexture("_Tex", textures[i]);
            index = i + 1; // 更新index（用于数据记录）
            pictureCount++; // 计数+1

            yield return new WaitForSeconds(waitTime); // 等待15s（主要图片显示时间）

            // 新增：每13张图片后，休息10s
            if (pictureCount % 13 == 0 && pictureCount < textures.Length)
            {
                isStartSave = false; // 暂停数据记录
                if (restText != null)
                {
                    restText.gameObject.SetActive(true); // 显示休息提示
                }
                yield return new WaitForSeconds(10f); // 休息10s
                if (restText != null)
                {
                    restText.gameObject.SetActive(false); // 隐藏休息提示
                }
                isStartSave = true; // 恢复数据记录
            }
        }

        // 结束后显示GameOver UI（原逻辑）
        transform.GetChild(1).gameObject.SetActive(true);
        SaveCSV(data);
        Debug.Log("saved");
    }

    void ShuffleTextures()
    {
        for (int i = 0; i < textures.Length; i++)
        {
            Cubemap temp = textures[i];
            int randomIndex = UnityEngine.Random.Range(i, textures.Length);
            textures[i] = textures[randomIndex];
            textures[randomIndex] = temp;
        }
    }

    void SaveCSV(List<string[]> data)
    {
        DateTime currentTime = DateTime.Now;
        string fileName = currentTime.ToString("yyyy_MM_dd HH_mm_ss") + ".csv";
        string filePath = Path.Combine(Application.persistentDataPath, fileName);

        using (StreamWriter writer = new StreamWriter(filePath, false))
        {
            foreach (var row in data)
            {
                writer.WriteLine(string.Join(",", row));
            }
        }

        Debug.Log("CSV located at " + filePath);
    }

    public Text debugText; // 如果原有，保持
    void Update()
    {
        RecordTrackData();
    }
}