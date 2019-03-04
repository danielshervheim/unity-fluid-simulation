using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CPUFluid2D : MonoBehaviour {

	public int n = 64;
	private int size;

	public bool showVelocity = false;

	public float diff = 0f;
	public float visc = 0f;
	public float force = 50f;
	public float source = 100f;

	private float[] u, u_prev, v, v_prev;
	private float[] dens, dens_prev;

	private Vector3 mousePos, mouseDelta;
	
	public Material material;
	private Texture2D texture;

	// Use this for initialization
	void Start () {
		size = (n+2)*(n+2);

		/* Create the empty arrays. */
		u = new float[size];
		u_prev = new float[size];

		v = new float[size];
		v_prev = new float[size];

		dens = new float[size];
		dens_prev = new float[size];

		/* Setup the camera to look over the simulation. This allows
		us to easily calculate the mouse position in simulation space. */
		Camera.main.orthographic = true;
		Camera.main.orthographicSize = (n+2)/2f;
		Camera.main.transform.position = new Vector3(0.5f*(n+2f), 10f, 0.5f*(n+2f));
		Camera.main.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

		/* Create Texture to display info. */
		texture = new Texture2D(n+2, n+2, TextureFormat.RGBAHalf, false);

		/* Instantiate and transform a plane to show the texture. */
		GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
		plane.transform.localScale = 0.1f * new Vector3(n+2f, 10f, n+2f);
		plane.transform.localPosition = 0.5f*(n+2f)*(Vector3.right+Vector3.forward);
		plane.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
		
		/* Assign the material to the plane, and the texture to the material. */
		plane.GetComponent<MeshRenderer>().material = material;
		material.SetTexture("_MainTex", texture);
	}
	
	// Update is called once per frame
	void Update () {
		// Update the mouse variables.
		mouseDelta = GetMousePos() - mousePos;
		mousePos = GetMousePos();

		if (Input.GetMouseButtonDown(2)) {
			u = new float[size];
			v = new float[size];
			dens = new float[size];
		}

		// Advance simulation.
		GetFromUI(ref dens_prev, ref u_prev, ref v_prev);
		VelocityStep(n, ref u, ref v, ref u_prev, ref v_prev, visc, Time.deltaTime);
		DensityStep(n, ref dens, ref dens_prev, ref u, ref v, diff, Time.deltaTime);

		// Upload the density as colors to the texture.
		Color[] cols = texture.GetPixels(0);
		for (int i = 0; i < cols.Length; i++) {
			if (!showVelocity) {
				cols[i] = new Color(dens[i], dens[i], dens[i], 1f);
			}
			else {
				cols[i] = new Color(u[i], v[i], 0f, 1f);
			}
		}
		texture.SetPixels(cols, 0);
		texture.Apply();
	}


	void Swap(ref float[] a, ref float[] b) {
		var tmp = a;
		a = b;
		b = tmp;
	}


	void AddSource(int n, ref float[] x, ref float[] s, float dt) {
		for (int i = 0; i < size; i++) {
			x[i] += dt*s[i];
		}
	}

	void Diffuse(int n, int b, ref float[] x, ref float[] x0, float diff, float dt) {
		float a = dt * diff * n * n;
		LinearSolve(n, b, ref x, ref x0, a, 1f+4f*a);
	}

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
	}

	void LinearSolve(int n, int b, ref float[] x, ref float[] x0, float a, float c) {
		int i, j, k;

		for (k = 0; k < 20; k++) {
			for (i = 1; i <= n; i++) {
				for (j = 1; j <= n; j++) {
					x[To1D(i,j)] = (x0[To1D(i,j)] + a*(x[To1D(i-1,j)]+x[To1D(i+1,j)]+x[To1D(i,j-1)]+x[To1D(i,j+1)]))/c;
				}
			}
			SetBoundary(n, b, ref x);
		}
	}

	
	void Project(int n, ref float[] u, ref float[] v, ref float[] p, ref float[] div) {
		int i, j;

		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) {
				div[To1D(i, j)] = -0.5f*(u[To1D(i+1,j)]-u[To1D(i-1,j)]+v[To1D(i,j+1)]-v[To1D(i,j-1)])/n;
				p[To1D(i, j)] = 0.0f;
			}
		}

		SetBoundary(n, 0, ref div);
		SetBoundary(n, 0, ref p);

		LinearSolve(n, 0, ref p, ref div, 1, 4);

		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) {
				u[To1D(i,j)] -= 0.5f * n * (p[To1D(i+1,j)] - p[To1D(i-1,j)]);
				v[To1D(i,j)] -= 0.5f * n * (p[To1D(i,j+1)] - p[To1D(i,j-1)]);
			}
		}

		SetBoundary(n, 1, ref u);
		SetBoundary(n, 2, ref v);
	}
	
	void Advect (int n, int b, ref float[] d, ref float[] d0, ref float[] u, ref float[] v, float dt) {
		int i, j, i0, j0, i1, j1;
		float x, y, s0, t0, s1, t1, dt0;

		dt0 = dt * n;

		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) {
				x = i - dt0*u[To1D(i,j)];
				y = j - dt0*v[To1D(i,j)];
				
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
				
				d[To1D(i,j)] = 	s0*(t0*d0[To1D(i0,j0)] + t1*d0[To1D(i0,j1)]) +
								s1*(t0*d0[To1D(i1,j0)] + t1*d0[To1D(i1,j1)]);
			}
		}

		SetBoundary(n, b, ref d);
	}




	void VelocityStep(int n, ref float[] u, ref float[] v, ref float[] u0, ref float[] v0, float visc, float dt) {
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



	void DensityStep(int n, ref float[] x, ref float[] x0, ref float[] u, ref float[] v, float diff, float dt) {
		AddSource(n, ref x, ref x0, dt);
		Swap(ref x0, ref x);
		Diffuse(n, 0, ref x, ref x0, diff, dt);
		Swap(ref x0, ref x);
		Advect(n, 0, ref x, ref x0, ref u, ref v, dt);
	}




	void GetFromUI(ref float[] d, ref float[] u, ref float[] v) {
		for (int i = 0; i < size; i++) {
			d[i] = u[i] = v[i] = 0f;
		}

		if (!Input.GetMouseButton(0) && !Input.GetMouseButton(1)) {
			return;
		}

		int x = GetIdFromPosition(mousePos).x;
		int y = GetIdFromPosition(mousePos).y;

		if (x < 1 || x > n || y < 1 || y > n) {
			return;
		}

		if (Input.GetMouseButton(0)) {
			u[To1D(x, y)] = force * mouseDelta.x;
			v[To1D(x, y)] = force * mouseDelta.z;
		}

		if (Input.GetMouseButton(1)) {
			d[To1D(x, y)] = source;
		}
	}






	Vector2Int To2D(int i) {
		return new Vector2Int(i%(n+2), i/(n+2));
	}



	int To1D(int i, int j) {
		return i + j*(n+2); 
	}

	/* Returns the mouse position in simulation space. */
	Vector3 GetMousePos() {
		Vector3 p = Camera.main.ScreenToWorldPoint(Input.mousePosition);
		return new Vector3(p.x, 0f, p.z);
	}

	/* Returns the ID of the cell nearest to the input position. */
	Vector2Int GetIdFromPosition(Vector3 p) {
		int x = (int)Mathf.Clamp(p.x, 0f, (n+2) - 1);
		int y = (int)Mathf.Clamp(p.z, 0f, (n+2) - 1);
		return new Vector2Int(x, y);
	}
}
