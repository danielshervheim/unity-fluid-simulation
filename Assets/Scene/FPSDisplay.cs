//
// Copyright © Daniel Shervheim, 2019
// www.danielshervheim.com
//

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FPSDisplay : MonoBehaviour {
	public GPUFluid2D gpuFluid2D;

	bool show = true;

	void Update() {
		if (Input.GetKeyUp(KeyCode.RightShift)) {
            show = !show;
        }
	}

	void OnGUI() {
		if (show) {
			GUI.Label(new Rect(25, 25, 100, 25), (gpuFluid2D.n+2) + " x" + (gpuFluid2D.n + 2) + " grid");
        	GUI.Label(new Rect(25, 50, 100, 25), 1.0f/Time.smoothDeltaTime + " fps");
		}  
    }
}
