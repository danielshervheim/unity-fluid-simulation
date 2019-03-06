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

	// Arrays to hold the field values.
	int size, n2, threadGroups;
	ComputeBuffer u, u_prev, v, v_prev, dens, dens_prev;

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
		u = new ComputeBuffer(size, 4);
		u.SetData(new float[size]);
		u_prev = new ComputeBuffer(size, 4);
		u_prev.SetData(new float[size]);

		v = new ComputeBuffer(size, 4);
		v.SetData(new float[size]);
		v_prev = new ComputeBuffer(size, 4);
		v_prev.SetData(new float[size]);

		dens = new ComputeBuffer(size, 4);
		dens.SetData(new float[size]);
		dens_prev = new ComputeBuffer(size, 4);
		dens_prev.SetData(new float[size]);

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
		compute.SetBuffer(kernels.Buffer2Texture, "b2t_buffer", dens);
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
		GetFromUI(ref dens_prev, ref u_prev, ref v_prev);
		VelocityStep(n, ref u, ref v, ref u_prev, ref v_prev, visc, Time.deltaTime);
		DensityStep(n, ref dens, ref dens_prev, ref u, ref v, diff, Time.deltaTime);

		// Upload the density as colors to the texture.
		DrawDensity();
	}



	void Swap(ref ComputeBuffer a, ref ComputeBuffer b) {
		var tmp = a;
		a = b;
		b = tmp;
	}



	void AddSource(int n, ref ComputeBuffer x, ref ComputeBuffer s, float dt) {
		compute.SetInt("as_n", n);
		compute.SetBuffer(kernels.AddSource, "as_x", x);
		compute.SetBuffer(kernels.AddSource, "as_s", s);
		compute.SetFloat("as_dt", dt);
		compute.Dispatch(kernels.AddSource, threadGroups, threadGroups, 1);

		// for (int i = 0; i < size; i++) {
		// 	x[i] += dt*s[i];
		// }
	}



	void Diffuse(int n, int b, ref ComputeBuffer x, ref ComputeBuffer x0, float diff, float dt) {
		float a = dt * diff * n * n;
		LinearSolve(n, b, ref x, ref x0, a, 1f+4f*a);
	}



	/*	// Note: we have chosen to omit the boundary conditions for now as
		// it was causing weird wraparound artefacts which I didn't feel like debugging
		// due to project due-date time crunch :)
	void SetBoundary(int n, int b, ref float[] x) {
		int i;

		for (i = 1; i <= n; i++) {
			x[To1D(0, i)] = b==1 ? -x[To1D(1, i)] : x[To1D(1, i)];
			x[To1D(n+1, i)] = b==1 ? -x[To1D(n, i)] : x[To1D(n, i)];
			x[To1D(i, 0)] = b==2 ? -x[To1D(i, 1)] : x[To1D(i, 1)];
			x[To1D(i, n+1)] = b==2 ? -x[To1D(i, n)] : x[To1D(i, n)];
		}

		x[To1D(0, 0)] = 0.5f * (x[To1D(1, 0)] + x[To1D(0, 1)]);
		x[To1D(0, n+1)] = 0.5f * (x[To1D(1, n+1)] + x[To1D(0, n)]);
		x[To1D(n+1, 0)] = 0.5f * (x[To1D(n, 0)] + x[To1D(n+1, 1)]);
		x[To1D(n+1, n+1)] = 0.5f * (x[To1D(n, n+1)] + x[To1D(n+1, n)]);
	} */



	void LinearSolve(int n, int b, ref ComputeBuffer x, ref ComputeBuffer x0, float a, float c) {
		compute.SetInt("ls_n", n);
		compute.SetInt("ls_b", b);
		compute.SetBuffer(kernels.LinearSolve, "ls_x", x);
		compute.SetBuffer(kernels.LinearSolve, "ls_x0", x0);
		compute.SetFloat("ls_a", a);
		compute.SetFloat("ls_c", c);
		compute.Dispatch(kernels.LinearSolve, threadGroups, threadGroups, 1);

		/*
		buffer1.SetData(x);
		buffer2.SetData(x0);

		compute.SetInt("ls_n", n);
		compute.SetInt("ls_b", b);
		compute.SetBuffer(k_linearSolve, "ls_x", buffer1);
		compute.SetBuffer(k_linearSolve, "ls_x0", buffer2);
		compute.SetFloat("ls_a", a);
		compute.SetFloat("ls_c", c);

		compute.Dispatch(k_linearSolve, (n+2)/32 + 1, (n+2)/32 + 1, 1);

		buffer1.GetData(x);
		buffer2.GetData(x0);
		*/
	}


	void Project(int n, ref ComputeBuffer u, ref ComputeBuffer v, ref ComputeBuffer p, ref ComputeBuffer div) {
		compute.SetInt("ps_n", n);
		compute.SetBuffer(kernels.ProjectStart, "ps_u", u);
		compute.SetBuffer(kernels.ProjectStart, "ps_v", v);
		compute.SetBuffer(kernels.ProjectStart, "ps_p", p);
		compute.SetBuffer(kernels.ProjectStart, "ps_div", div);
		compute.Dispatch(kernels.ProjectStart, threadGroups, threadGroups, 1);

		LinearSolve(n, 0, ref p, ref div, 1f, 4f);

		compute.SetInt("pf_n", n);
		compute.SetBuffer(kernels.ProjectFinish, "pf_u", u);
		compute.SetBuffer(kernels.ProjectFinish, "pf_v", v);
		compute.SetBuffer(kernels.ProjectFinish, "pf_p", p);
		compute.Dispatch(kernels.ProjectFinish, threadGroups, threadGroups, 1);

		/*
		buffer1.SetData(u);
		buffer2.SetData(v);
		buffer3.SetData(p);
		buffer4.SetData(div);

		compute.SetInt("proj_1_n", n);
		compute.SetBuffer(k_project1, "proj_1_u", buffer1);
		compute.SetBuffer(k_project1, "proj_1_v", buffer2);
		compute.SetBuffer(k_project1, "proj_1_p", buffer3);
		compute.SetBuffer(k_project1, "proj_1_div", buffer4);

		compute.Dispatch(k_project1, (n+2)/32 + 1, (n+2)/32 + 1, 1);

		buffer1.GetData(u);
		buffer2.GetData(v);
		buffer3.GetData(p);
		buffer4.GetData(div);

		LinearSolve(n, 0, ref p, ref div, 1f, 4f);

		buffer1.SetData(u);
		buffer2.SetData(v);
		buffer3.SetData(p);
		buffer4.SetData(div);

		compute.SetInt("proj_2_n", n);
		compute.SetBuffer(k_project2, "proj_2_u", buffer1);
		compute.SetBuffer(k_project2, "proj_2_v", buffer2);
		compute.SetBuffer(k_project2, "proj_2_p", buffer3);
		compute.SetBuffer(k_project2, "proj_2_div", buffer4);

		compute.Dispatch(k_project2, (n+2)/32 + 1, (n+2)/32 + 1, 1);

		buffer1.GetData(u);
		buffer2.GetData(v);
		buffer3.GetData(p);
		buffer4.GetData(div);
		*/
	}

	void Advect (int n, int b, ref ComputeBuffer d, ref ComputeBuffer d0, ref ComputeBuffer u, ref ComputeBuffer v, float dt) {
		compute.SetInt("ad_n", n);
		compute.SetInt("ad_b", b);
		compute.Setbuffer(kernels.Advect, "ad_d", d);
		compute.Setbuffer(kernels.Advect, "ad_d0", d0);
		compute.Setbuffer(kernels.Advect, "ad_u", u);
		compute.Setbuffer(kernels.Advect, "ad_v", v);
		compute.SetFloat("ad_dt", dt);
		compute.Dispatch(kernels.Advect, threadGroups, threadGroups, 1);

		/*
		buffer1.SetData(d);
		buffer2.SetData(d0);
		buffer3.SetData(u);
		buffer4.SetData(v);

		compute.SetInt("adv_n", n);
		compute.SetInt("adv_b", b);
		compute.SetBuffer(k_advect, "adv_d", buffer1);
		compute.SetBuffer(k_advect, "adv_d0", buffer2);
		compute.SetBuffer(k_advect, "adv_u", buffer3);
		compute.SetBuffer(k_advect, "adv_v", buffer4);
		compute.SetFloat("adv_dt", dt);

		compute.Dispatch(k_advect, (n+2)/32 + 1, (n+2)/32 + 1, 1);

		buffer1.GetData(d);
		buffer2.GetData(d0);
		buffer3.GetData(u);
		buffer4.GetData(v);
		*/
	}



	void VelocityStep(int n, ref ComputeBuffer u, ref ComputeBuffer v, ref ComputeBuffer u0, ref ComputeBuffer v0, float visc, float dt) {
		AddSource(n, ref u, ref u0, dt);
		AddSource(n, ref v, ref v0, dt);

		Swap(ref u0, ref u);
		Diffuse(n, 1, ref u, ref u0, visc, dt);

		Swap(ref v0, ref v);
		Diffuse(n, 2, ref v, ref v0, visc, dt);

		Project(n, ref u, ref v, ref u0, ref v0);

		Swap(ref u0, ref u);
		Swap(ref v0, ref v);

		Advect(n, 1, ref u, ref u0, ref u0, ref v0, dt);
		Advect(n, 2, ref v, ref v0, ref u0, ref v0, dt);

		Project(n, ref u, ref v, ref u0, ref v0);
	}



	void DensityStep(int n, ref ComputeBuffer x, ref ComputeBuffer x0, ref ComputeBuffer u, ref ComputeBuffer v, float diff, float dt) {
		AddSource(n, ref x, ref x0, dt);
		Swap(ref x0, ref x);
		Diffuse(n, 0, ref x, ref x0, diff, dt);
		Swap(ref x0, ref x);
		Advect(n, 0, ref x, ref x0, ref u, ref v, dt);
	}



	void GetFromUI(ref ComputeBuffer d, ref ComputeBuffer u, ref ComputeBuffer v) {
		d.SetData(new float[size]);
		u.SetData(new float[size]);
		v.SetData(new float[size]);

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
			uArray[To1D(x, y)] = force * mouseDelta.x * (n+2);  // scale by resolution.
			u.SetData(uArray);

			float[] vArray = new float[size];
			vArray[To1D(x, y)] = force * mouseDelta.z * (n+2);  // scale by resolution.
			v.SetData(vArray);
		}

		if (Input.GetMouseButton(1)) {
			float[] dArray = new float[size];
			dArray[To1D(x, y)] = source;
			d.SetData(dArray);
		}
	}



	void ClearFields() {
		u.SetData(new float[size]);
		v.SetData(new float[size]);
		dens.SetData(new float[size]);
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
		if (u != null) {
			u.Release();
		}
		if (u_prev != null) {
			u_prev.Release();
		}
		if (v != null) {
			v.Release();
		}
		if (v_prev != null) {
			v_prev.Release();
		}
		if (dens != null) {
			dens.Release();
		}
		if (dens_prev != null) {
			dens_prev.Release();
		}

		if (texture != null) {
			texture.Release();
		}
	}
}
