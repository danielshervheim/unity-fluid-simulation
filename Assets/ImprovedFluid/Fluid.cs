using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fluid : MonoBehaviour {
	[Header("Required")]
	public Material material;
	public ComputeShader compute;

	[Header("Simulation Parameters")]
	public int n = 64;
	public float diff = 0f;
	public float visc = 0f;
	public float force = 75f;
	public float source = 100f;

	int size, n2, threadGroups;

	// Compute buffers and indices to hold the fields.
	ComputeBuffer[] buffers;
	int u, u_prev, v, v_prev, dens, dens_prev;

	// Texture to visualize the density field.
	RenderTexture texture;

	// Compute kernels.
	private struct Kernels {
		public int AddSource, LinearSolve, ProjectStart, ProjectFinish, Advect, Buffer2Texture;
	}
	Kernels kernels;

	// Mouse movement variables.
	Vector3 mousePos, mouseDelta;

	void Start () {
		n2 = n+2;
		size = n2*n2;
		threadGroups = n2 / 32 + 1;

		// Create the empty buffers and set them to zero.
		u = 0; u_prev = 1; v = 2; v_prev = 3; dens = 4; dens_prev = 5;

		buffers = new ComputeBuffer[6];

		buffers[u] = new ComputeBuffer(size, 4);
		buffers[u].SetData(new float[size]);
		buffers[u_prev] = new ComputeBuffer(size, 4);
		buffers[u_prev].SetData(new float[size]);

		buffers[v] = new ComputeBuffer(size, 4);
		buffers[v].SetData(new float[size]);
		buffers[v_prev] = new ComputeBuffer(size, 4);
		buffers[v_prev].SetData(new float[size]);

		buffers[dens] = new ComputeBuffer(size, 4);
		buffers[dens].SetData(new float[size]);
		buffers[dens_prev] = new ComputeBuffer(size, 4);
		buffers[dens_prev].SetData(new float[size]);

		// Spawn a plane to render the field(s) onto.
		GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
		plane.transform.parent = this.transform;
		plane.transform.localPosition = Vector3.zero;
		plane.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
		plane.transform.localScale = Vector3.one * 0.1f;
		plane.GetComponent<MeshRenderer>().material = material;

		// Adjust the camera to be over the plane.
		Camera.main.transform.parent = this.transform;
		Camera.main.transform.localPosition = Vector3.up * 5f;
		Camera.main.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
		Camera.main.transform.localScale = Vector3.one;
		Camera.main.orthographic = true;
		Camera.main.orthographicSize = 0.5f;

		// Create a new renderTexture to display the density field and assign it to the material.
		texture = new RenderTexture(n2, n2, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear);
		texture.enableRandomWrite = true;
		texture.Create();
		material.SetTexture("_MainTex", texture);

		// Get the compute kernels.
		kernels.AddSource = compute.FindKernel("AddSource");
		kernels.LinearSolve = compute.FindKernel("LinearSolve");
		kernels.ProjectStart = compute.FindKernel("ProjectStart");
		kernels.ProjectFinish = compute.FindKernel("ProjectFinish");
		kernels.Advect = compute.FindKernel("Advect");
		kernels.Buffer2Texture = compute.FindKernel("Buffer2Texture");

		// Set the n parameter for the compute shader.
		compute.SetInt("n", n);
		compute.SetInt("n2", n2);

		// Set the draw buffers and texture, which do not have to change at runtime.
		compute.SetBuffer(kernels.Buffer2Texture, "b2t_u", buffers[u]);
		compute.SetBuffer(kernels.Buffer2Texture, "b2t_v", buffers[v]);
		compute.SetBuffer(kernels.Buffer2Texture, "b2t_dens", buffers[dens]);
		compute.SetTexture(kernels.Buffer2Texture, "b2t_texture", texture);
	}



	void Update () {
		// Update the mouse variables.
		mouseDelta = GetMousePos() - mousePos;
		mousePos = GetMousePos();

		// Clear the fields if the middle mouse button is pressed.
		if (Input.GetMouseButtonDown(2)) {
			ClearFields();
		}

		// Set the delta time.
		compute.SetFloat("dt", Time.deltaTime);

		// Advance simulation.
		GetFromUI(dens_prev, u_prev, v_prev);
		VelocityStep(n, u, v, u_prev, v_prev, visc, Time.deltaTime);
		DensityStep(n, dens, dens_prev, u, v, diff, Time.deltaTime);

		// Upload the density as colors to the texture.
		DrawDensity();
	}



	void Swap(ref int a, ref int b) {
		int tmp = a;
		a = b;
		b = tmp;
	}



	void AddSource(int n, int x, int s, float dt) {
		compute.SetInt("as_n", n);
		compute.SetBuffer(kernels.AddSource, "as_x", buffers[x]);
		compute.SetBuffer(kernels.AddSource, "as_s", buffers[s]);
		compute.SetFloat("as_dt", dt);
		compute.Dispatch(kernels.AddSource, threadGroups, threadGroups, 1);
	}



	void Diffuse(int n, int b, int x, int x0, float diff, float dt) {
		float a = dt * diff * Mathf.Pow(n, 2f);
		LinearSolve(n, b, x, x0, a, 1f+4f*a);
	}



	void LinearSolve(int n, int b, int x, int x0, float a, float c) {
		compute.SetInt("ls_n", n);
		compute.SetInt("ls_b", b);
		compute.SetBuffer(kernels.LinearSolve, "ls_x", buffers[x]);
		compute.SetBuffer(kernels.LinearSolve, "ls_x0", buffers[x0]);
		compute.SetFloat("ls_a", a);
		compute.SetFloat("ls_c", c);
		compute.Dispatch(kernels.LinearSolve, threadGroups, threadGroups, 1);
	}



	void Project(int n, int u, int v, int p, int div) {
		compute.SetInt("ps_n", n);
		compute.SetBuffer(kernels.ProjectStart, "ps_u", buffers[u]);
		compute.SetBuffer(kernels.ProjectStart, "ps_v", buffers[v]);
		compute.SetBuffer(kernels.ProjectStart, "ps_p", buffers[p]);
		compute.SetBuffer(kernels.ProjectStart, "ps_div", buffers[div]);
		compute.Dispatch(kernels.ProjectStart, threadGroups, threadGroups, 1);

		LinearSolve(n, 0, p, div, 1f, 4f);

		compute.SetInt("pf_n", n);
		compute.SetBuffer(kernels.ProjectFinish, "pf_u", buffers[u]);
		compute.SetBuffer(kernels.ProjectFinish, "pf_v", buffers[v]);
		compute.SetBuffer(kernels.ProjectFinish, "pf_p", buffers[p]);
		compute.Dispatch(kernels.ProjectFinish, threadGroups, threadGroups, 1);
	}



	void Advect (int n, int b, int d, int d0, int u, int v, float dt, bool gpu) {
		if (gpu) {
			compute.SetInt("ad_n", n);
			compute.SetInt("ad_b", b);
			compute.SetBuffer(kernels.Advect, "ad_d", buffers[d]);
			compute.SetBuffer(kernels.Advect, "ad_d0", buffers[d0]);
			compute.SetBuffer(kernels.Advect, "ad_u", buffers[u]);
			compute.SetBuffer(kernels.Advect, "ad_v", buffers[v]);
			compute.SetFloat("ad_dt", dt);
			compute.Dispatch(kernels.Advect, threadGroups, threadGroups, 1);
		}
		else {
			float[] td, td0, tu, tv;
			td = new float[size];
			td0 = new float[size];
			tu = new float[size];
			tv = new float[size]; 
			buffers[d].GetData(td);
			buffers[d0].GetData(td0);
			buffers[u].GetData(tu);
			buffers[v].GetData(tv); 

			int i, j, i0, j0, i1, j1;
			float x, y, s0, t0, s1, t1, dt0;

			dt0 = dt * n;

			for (i = 1; i <= n; i++) {
				for (j = 1; j <= n; j++) {
					x = i - dt0*tu[To1D(i,j)];
					y = j - dt0*tv[To1D(i,j)];
					
					if (x < 0.5f) x = 0.5f;
					if (x > n+0.5f) x = n+0.5f;
					i0 = (int)x;
					i1 = i0 + 1;
					
					if (y < 0.5f) y = 0.5f;
					if (y > n+0.5f) y = n + 0.5f;
					j0 = (int)y;
					j1 = j0 + 1;

					s1 = x - i0;
					s0 = 1f - s1;

					t1 = y - j0;
					t0 = 1f - t1;
					
					td[To1D(i,j)] =	s0*(t0*td0[To1D(i0,j0)] + t1*td0[To1D(i0,j1)]) +
									s1*(t0*td0[To1D(i1,j0)] + t1*td0[To1D(i1,j1)]);
				}
			}

			buffers[d].SetData(td);
			buffers[d0].SetData(td0);
			buffers[u].SetData(tu);
			buffers[v].SetData(tv); 
		}
	}



	void VelocityStep(int n, int u, int v, int u0, int v0, float visc, float dt) {
		AddSource(n, u, u0, dt);
		AddSource(n, v, v0, dt);

		Swap(ref u0, ref u);
		Diffuse(n, 1, u, u0, visc, dt);

		Swap(ref v0, ref v);
		Diffuse(n, 2, v, v0, visc, dt);

		Project(n, u, v, u0, v0);

		Swap(ref u0, ref u);
		Swap(ref v0, ref v);

		// BUG: these advections don't work on the GPU, but the density one does???

		Advect(n, 1, u, u0, u0, v0, dt, false);
		Advect(n, 2, v, v0, u0, v0, dt, false);

		Project(n, u, v, u0, v0);
	}



	void DensityStep(int n, int x, int x0, int u, int v, float diff, float dt) {
		AddSource(n, x, x0, dt);
		Swap(ref x0, ref x);
		Diffuse(n, 0, x, x0, diff, dt);
		Swap(ref x0, ref x);
		Advect(n, 0, x, x0, u, v, dt, true);
	}



	void GetFromUI(int d, int u, int v) {
		buffers[d].SetData(new float[size]);
		buffers[u].SetData(new float[size]);
		buffers[v].SetData(new float[size]);

		if (!Input.GetMouseButton(0) && !Input.GetMouseButton(1)) {
			return;
		}

		int x = GetIdFromPosition(mousePos).x;
		int y = GetIdFromPosition(mousePos).y;

		if (x < 1 || x > n || y < 1 || y > n) {
			return;
		}

		if (Input.GetMouseButton(0)) {
			float[] uArray = new float[size];
			uArray[To1D(x, y)] = force * mouseDelta.x * n2;  // scale by resolution.
			buffers[u].SetData(uArray);

			float[] vArray = new float[size];
			vArray[To1D(x, y)] = force * mouseDelta.z * n2;  // scale by resolution.
			buffers[v].SetData(vArray);
		}

		if (Input.GetMouseButton(1)) {
			float[] dArray = new float[size];
			dArray[To1D(x, y)] = source;
			buffers[d].SetData(dArray);
		}
	}



	void ClearFields() {
		buffers[u].SetData(new float[size]);
		buffers[v].SetData(new float[size]);
		buffers[dens].SetData(new float[size]);
	}



	void DrawDensity() {
		compute.Dispatch(kernels.Buffer2Texture, threadGroups, threadGroups, 1);
	}



	Vector2Int To2D(int i) {
		return new Vector2Int(i%n2, i/n2);
	}



	int To1D(int i, int j) {
		return i + j*n2;
	}



	/* Returns the mouse position in simulation space. */
	Vector3 GetMousePos() {
		Vector3 p = Camera.main.ScreenToWorldPoint(Input.mousePosition);
		return new Vector3(p.x, 0f, p.z);
	}



	/* Returns the ID of the cell nearest to the input position. */
	Vector2Int GetIdFromPosition(Vector3 p) {
		p += 0.5f * Vector3.one;
		p *= n2;
		return new Vector2Int((int)p.x, (int)p.z);
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
}
