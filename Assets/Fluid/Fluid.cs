//
// Copyright © Daniel Shervheim, 2019
// danielshervheim@gmail.com
// danielshervheim.com
//

using UnityEngine;

public class Fluid : MonoBehaviour {
    [Header("Required Assets")]
    public ComputeShader compute;
    public Material material;

    [Header("Preset Parameters")]
    public int n;
    int n2, len;

    [Header("Realtime Parameters")]
    public float diff = 0f;
	public float visc = 0f;
	public float force = 75f;
    public float source = 100f;

    // Kernels.
    int[] kernels;
    int addSource = 0;
    int linearSolve = 1;
    int advect = 2;
    int projectStart = 3;
    int projectFinish = 4;
    int bufferToTexture = 5;
    int clearBuffer = 6;

    // Buffers.
    ComputeBuffer[] buffers;
    int u = 0;
    int u0 = 1;
    int v = 2;
    int v0 = 3;
    int d = 4;
    int d0 = 5;

    // Texture.
    RenderTexture texture;

    // Mouse movement variables.
    Vector3 mousePos, mouseDelta;

    void Start() {
        // Calculate the global variables.
        n2 = n + 2;
        len = (int)Mathf.Pow(n2, 2f);

        // Set the global compute variables.
        compute.SetInt("n", n);
        compute.SetInt("n2", n2);
        compute.SetInt("len", len);

        // Find and assign the kernels.
        kernels = new int[7];
        kernels[addSource] = compute.FindKernel("AddSource");
        kernels[linearSolve] = compute.FindKernel("LinearSolve");
        kernels[advect] = compute.FindKernel("Advect");
        kernels[projectStart] = compute.FindKernel("ProjectStart");
        kernels[projectFinish] = compute.FindKernel("ProjectFinish");
        kernels[bufferToTexture] = compute.FindKernel("BufferToTexture");
        kernels[clearBuffer] = compute.FindKernel("ClearBuffer");

        // Create and empty the buffers.
        buffers = new ComputeBuffer[6];
        for (int b = 0; b < buffers.Length; b++) {
            buffers[b] = new ComputeBuffer(len, 4);
            ClearBuffer(b);
        }

        // Create the texture, and assign it to the material.
        texture = new RenderTexture(n2, n2, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear);
        texture.enableRandomWrite = true;
        texture.Create();
        material.SetTexture("_MainTex", texture);

        // Upload the texture and buffers to the gpu for rendering each frame.
        // Note: this is set only once in the Start() method because we will only ever
        // render these buffers in the bufferToTexture kernel.
        compute.SetTexture(kernels[bufferToTexture], "b2t_texture", texture);
        compute.SetBuffer(kernels[bufferToTexture], "b2t_u", buffers[u]);
        compute.SetBuffer(kernels[bufferToTexture], "b2t_v", buffers[v]);
        compute.SetBuffer(kernels[bufferToTexture], "b2t_d", buffers[d]);

        // Create a plane to render the display texture onto, and set its material.
        GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
		plane.transform.parent = this.transform;
		plane.transform.localPosition = Vector3.zero;
		plane.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
		plane.transform.localScale = Vector3.one * 0.1f;
		plane.GetComponent<MeshRenderer>().material = material;

		// Adjust the camera to be centered over the plane.
		Camera.main.transform.parent = this.transform;
		Camera.main.transform.localPosition = Vector3.up * 5f;
		Camera.main.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
		Camera.main.transform.localScale = Vector3.one;
		Camera.main.orthographic = true;
        Camera.main.orthographicSize = 0.5f;
    }

    void Update() {
        // Set the delta time on the GPU.
        compute.SetFloat("dt", Time.deltaTime);
        compute.SetFloat("diff", diff);
        compute.SetFloat("visc", visc);

        // Update the mouse variables.
		mouseDelta = GetMousePosition() - mousePos;
        mousePos = GetMousePosition();

        // Reset the simulation if the middle mouse button is pressed.
        if (Input.GetMouseButtonDown(2)) {
            ClearBuffer(u);
            ClearBuffer(v);
            ClearBuffer(d);
        }

        // Get the forces from the mouse, and store them in the temporary buffers.
        GetFromUI(d0, u0, v0);

        VelocityStep(u, v, u0, v0);

        DensityStep(d, d0, u, v);

        // Copy the buffers over to the texture to display them.
        compute.Dispatch(kernels[bufferToTexture], n2/32 + 1, n2/32 + 1, 1);
    }

    void OnDestroy() {
        for (int i = 0; i < buffers.Length; i++) {
            if (buffers[i] != null) {
                buffers[i].Release();
            }
        }

        if (texture != null) {
            texture.Release();
        }
    }



    //
    // SOLVER STEPS
    //

    // Get the forces from the mouse and update the fields.
    void GetFromUI(int d, int u, int v) {
        // Clear the buffers.
        ClearBuffer(d);
        ClearBuffer(u);
        ClearBuffer(v);

        // Get the buffer index based on the mouse position.
        int index = GetIndexFromMousePosition();

        int x = index%n2;
        int y = index/n2;

        if (!(x >= 1 && x <= n && y >= 1 && y <= n)) return;

        // Add a force to the velocity field based on left click + drag.
        if (Input.GetMouseButton(0)) {
            float[] tu = new float[len];
            tu[index] = force * mouseDelta.x * n2;
            buffers[u].SetData(tu, index, index, 1);

            float[] tv = new float[len];
            tv[index] = force * mouseDelta.z * n2;
            buffers[v].SetData(tv, index, index, 1);
        }

        // Add more density to the density field based on right click.
        if (Input.GetMouseButton(1)) {
            float[] td = new float[len];
            td[index] = source;
            buffers[d].SetData(td, index, index, 1);
        }
    }

    // Update the velocity field to its next state.
    void VelocityStep(int u, int v, int u0, int v0) {
        AddSource(u, u0);
        AddSource(v, v0);

        Swap(ref u, ref u0);
        Diffuse(u, u0);

        Swap(ref v, ref v0);
        Diffuse(v, v0);

        Project(u, v, u0, v0);

        Swap(ref u0, ref u);
        Swap(ref v0, ref v);

        Advect(u, u0, u0, v0);
        Advect(v, v0, u0, v0);

        Project(u, v, u0, v0);
    }

    // Update the density field to its next state.
    void DensityStep(int x, int x0, int u, int v) {
        AddSource(x, x0);

        Swap(ref x, ref x0);
        Diffuse(x, x0);

        Swap(ref x, ref x0);
        Advect(x, x0, u, v);
    }



    //
    // SOLVER METHODS
    //

    void AddSource(int x, int s) {
        compute.SetBuffer(kernels[addSource], "as_x", buffers[x]);
        compute.SetBuffer(kernels[addSource], "as_s", buffers[s]);
        compute.Dispatch(kernels[addSource], len/256+1, 1, 1);
    }

    void Diffuse(int x, int x0) {
        float a = Time.deltaTime * diff * Mathf.Pow(n, 2f);
        LinearSolve(x, x0, a, 1f + 4f*a);
    }

    void Project(int u, int v, int p, int div) {
        compute.SetBuffer(kernels[projectStart], "ps_u", buffers[u]);
        compute.SetBuffer(kernels[projectStart], "ps_v", buffers[v]);
        compute.SetBuffer(kernels[projectStart], "ps_p", buffers[p]);
        compute.SetBuffer(kernels[projectStart], "ps_div", buffers[div]);
        compute.Dispatch(kernels[projectStart], len/256+1, 1, 1);

        LinearSolve(p, div, 1f, 4f);

        compute.SetBuffer(kernels[projectFinish], "pf_u", buffers[u]);
        compute.SetBuffer(kernels[projectFinish], "pf_v", buffers[v]);
        compute.SetBuffer(kernels[projectFinish], "pf_p", buffers[p]);
        compute.Dispatch(kernels[projectFinish], len/256+1, 1, 1);
    }

    void Advect(int d, int d0, int u, int v) {
    	compute.SetBuffer(kernels[advect], "ad_d", buffers[d]);
	    compute.SetBuffer(kernels[advect], "ad_d0", buffers[d0]);
	    compute.SetBuffer(kernels[advect], "ad_u", buffers[u]);
	    compute.SetBuffer(kernels[advect], "ad_v", buffers[v]);
	    compute.Dispatch(kernels[advect], len/256+1, 1, 1);
    }

    void LinearSolve(int x, int x0, float a, float c) {
    	compute.SetBuffer(kernels[linearSolve], "ls_x", buffers[x]);
        compute.SetBuffer(kernels[linearSolve], "ls_x0", buffers[x0]);
        compute.SetFloat("ls_a", a);
        compute.SetFloat("ls_c", c);
        for (int k = 0; k < 20; k++) {
            compute.Dispatch(kernels[linearSolve], len/256 + 1, 1, 1);
        }
    }



    //
    // UTILITIES
    //

    // Get the mouse position in world space orthographic view, top down (x, z).
    Vector3 GetMousePosition() {
        Vector3 tmp = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        return new Vector3(tmp.x, 0f, tmp.z);
    }

    // Returns the index of the grid which the mouse position is currently over.
    int GetIndexFromMousePosition() {
        Vector3 tmp = mousePos;  // -orthoSize:orthoSize
        tmp += Camera.main.orthographicSize*new Vector3(1f, 0f, 1f);  // 0:1
        tmp *= n2;  // 0:n2
        int x = (int)Mathf.Clamp(Mathf.Floor(tmp.x), 0f, n2-1);
        int y = (int)Mathf.Clamp(Mathf.Floor(tmp.z), 0f, n2-1);
        return y*n2 + x;
    }

    // Swaps two given integers.
    void Swap(ref int a, ref int b) {
        var tmp = a;
        a = b;
        b = tmp;
    }

    // Clears the given buffer to 0.0.
    void ClearBuffer(int x) {
        compute.SetBuffer(kernels[clearBuffer], "cb_buffer", buffers[x]);
        compute.Dispatch(kernels[clearBuffer], len/256+1, 1, 1);
    }
}
