using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fluid : MonoBehaviour {

	public int n = 256;
	public bool debugMouse = true;

	[Header("Velocity Field")]
	public bool debugVelocity = true;
	public bool affectVelocity = true;
	public int velocityAffectionRadius = 5;

	[Header("Density Field")]
	public bool debugDensity = true;
	public bool affectDensity = true;
	public bool increaseDensity = true;
	public int densityAffectionRadius = 5;

	// Velocity arrays.
	private Vector2[,] velocity;
	private Vector2[,] velocityPrevious;

	// Density arrays. (Density is a scalar 0:1 parameter).
	private float[,] density;
	private float[,] densityPrevious;

	// Housekeeping.
	private bool mouseDown = false;
	private Vector3 mousePosPrev;



	void Start () {
		velocity = new Vector2[n, n];
		velocityPrevious = new Vector2[n, n];
		density = new float[n, n];
		densityPrevious = new float[n, n];

		/* Setup the camera to look over the simulation. This allows
		us to easily calculate the mouse position in simulation space. */
		Camera.main.orthographic = true;
		Camera.main.orthographicSize = n/2 + 1;
		Camera.main.transform.position = new Vector3(n/2, 10f, n/2);
		Camera.main.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
	}



	void Update () {
		// Mouse down.
		if (Input.GetMouseButtonDown(0)) {
			mouseDown = true;
			mousePosPrev = GetMousePos();
		}

		// Mouse drag.
		if (Input.GetMouseButton(0)) {
			if (mouseDown) {
				Vector3 mousePos = GetMousePos();
				Vector3 mouseDelta = mousePos - mousePosPrev;
				mousePosPrev = mousePos;

				if (debugMouse) {
					Debug.DrawLine(mousePos, mousePos+mouseDelta, Color.white, 0f, true);
				}

				if (affectVelocity) {
					UpdateVelocityField(mousePos, mouseDelta);
				}

				if (affectDensity) {
					UpdateDensityField(mousePos, mouseDelta);
				}
			}
		}

		// Mouse up.
		if (Input.GetMouseButtonUp(0)) {
			mouseDown = false;
		}

		// Advance the simulation.

		// todo: simulation code!
	}



	/* Returns the mouse position in simulation space. */
	Vector3 GetMousePos() {
		Vector3 p = Camera.main.ScreenToWorldPoint(Input.mousePosition);
		return new Vector3(p.x, 0f, p.z);
	}



	/* Returns the ID of the cell nearest to the input position. */
	Vector2Int GetIdFromPosition(Vector3 p) {
		int x = (int)Mathf.Clamp(p.x, 0f, n-1);
		int y = (int)Mathf.Clamp(p.z, 0f, n-1);
		return new Vector2Int(x, y);
	}



	/* Updates the velocity field as a kernel around the mouse position, based on
	the mouse movement. */
	void UpdateVelocityField(Vector3 mousePos, Vector3 mouseDelta) {
		Vector2Int id = GetIdFromPosition(mousePos);
		for (int y = id.y - velocityAffectionRadius; y < id.y + velocityAffectionRadius; y++) {
			for (int x = id.x - velocityAffectionRadius; x < id.x + velocityAffectionRadius; x++) {
				if (x >= 0 && x < n-1 && y >= 0 && y < n-1) {
					Vector2Int tmp = new Vector2Int(x, y);
					float d = Vector2.Distance((Vector2)tmp, (Vector2)id) / (float)velocityAffectionRadius;
					d = 1.0f - Mathf.Clamp(d, 0f, 1f);
				
					velocity[tmp.x, tmp.y] += new Vector2(mouseDelta.x, mouseDelta.z) * d;

					// velocity[tmp.x, tmp.y] = Vector2.ClampMagnitude(velocity[tmp.x, tmp.y], 1.0f);
				}
			}
		}
	}



	/* Updates the density field as a kernel around the mouse position.*/
	void UpdateDensityField(Vector3 mousePos, Vector3 mouseDelta) {
		Vector2Int id = GetIdFromPosition(mousePos);
		for (int y = id.y - densityAffectionRadius; y < id.y + densityAffectionRadius; y++) {
			for (int x = id.x - densityAffectionRadius; x < id.x + densityAffectionRadius; x++) {
				if (x >= 0 && x < n-1 && y >= 0 && y < n-1) {
					Vector2Int tmp = new Vector2Int(x, y);
					float d = Vector2.Distance((Vector2)tmp, (Vector2)id) / (float)densityAffectionRadius;
					d = 1.0f - Mathf.Clamp(d, 0f, 1f);
				
					density[tmp.x, tmp.y] += d*(increaseDensity ? 1f : -1f);

					density[tmp.x, tmp.y] = Mathf.Clamp(density[tmp.x, tmp.y], 0f, 1f);
				}
			}
		}
	}



	void OnDrawGizmos() {
		// Draw the velocity field.
		if (velocity != null && debugVelocity) {
			Gizmos.color = Color.blue;
			for (int y = 0; y < n; y++) {
				for (int x = 0; x < n; x++) {
					Vector3 cell = new Vector3((float)x, 0f, (float)y);
					Vector3 dir = new Vector3(velocity[x, y].x, 0f, velocity[x, y].y);
					Gizmos.DrawLine(cell, cell+dir);
				}
			}
		}

		// Draw the density field.
		if (density != null && debugDensity) {
			Gizmos.color = Color.black;
			for (int y = 0; y < n; y++) {
				for (int x = 0; x < n; x++) {
					Vector3 cell = new Vector3((float)x, 0f, (float)y);
					Gizmos.DrawSphere(cell, density[x, y]);
				}
			}
		}
	}
}
