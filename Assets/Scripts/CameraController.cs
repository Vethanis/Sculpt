using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using static UnityEngine.Input;

public class CameraController : MonoBehaviour
{
    Transform m_xform;

    public float m_linearSpeed = 5.0f;
    public float m_angularSpeed = 10.0f;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        m_xform = GetComponent<Camera>().transform;
    }
    
    void Update()
    {
        float dt = Time.deltaTime;

        if(GetKey(KeyCode.Escape))
        {
            Application.Quit();
        }
        if(GetKey(KeyCode.W))
        {
            m_xform.position += m_xform.forward * m_linearSpeed * dt;
        }
        if(GetKey(KeyCode.S))
        {
            m_xform.position -= m_xform.forward * m_linearSpeed * dt;
        }
        if (GetKey(KeyCode.D))
        {
            m_xform.position += m_xform.right * m_linearSpeed * dt;
        }
        if (GetKey(KeyCode.A))
        {
            m_xform.position -= m_xform.right * m_linearSpeed * dt;
        }
        if(GetKey(KeyCode.Space))
        {
            m_xform.position += m_xform.up * m_linearSpeed * dt;
        }
        if (GetKey(KeyCode.LeftShift))
        {
            m_xform.position -= m_xform.up * m_linearSpeed * dt;
        }

        float dx = GetAxis("Mouse X") * m_angularSpeed * dt;
        float dy = GetAxis("Mouse Y") * m_angularSpeed * dt;

        m_xform.Rotate(Vector3.right, -dy);
        m_xform.Rotate(Vector3.up, dx);
        m_xform.LookAt(m_xform.position + m_xform.forward, Vector3.up);
    }
}
