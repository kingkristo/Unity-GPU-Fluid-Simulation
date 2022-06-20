using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FluidSimMain : MonoBehaviour
{
    public  FluidSim fluidSim;

    void Start()
    {
        fluidSim.Init();
    }

    void OnDisable()
    {
        fluidSim.ReleaseResources();
    }

    //blit to screen
    void OnRenderImage(RenderTexture source, RenderTexture destination) {
        fluidSim.RenderImage(destination);
    }

    void Update()
    {
        fluidSim.Tick();
    }
}
