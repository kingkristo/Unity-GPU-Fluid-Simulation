using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[System.Serializable]
public class FluidSim
{
    //shader
    public ComputeShader fluidShader;
    //buffers
    private ComputeBuffer velocityBuffer;                   
    private ComputeBuffer divergenceBuffer;            
    private ComputeBuffer pressureBuffer;               
    private ComputeBuffer tempBuffer;
    private ComputeBuffer inkBuffer;
    //vars
    private RenderTexture outTexture;
    private Vector2 oldMousePos;
    private int jacobiIterations = 100; //[NOTE: This needs to be an even number or will break]
    private int gridResolution = 672;
    private float viscosity = 0.01f;
    private float inkDissipation = 0.99f;
    private float velocityDissipation = 0.999f;
    //index for kernels, hold them here instead of recalculating to save time
    private int copyBufferKernel;
    private int clearBufferKernel;
    private int advectionKernel;
    private int divergenceKernel;
    private int jacobiPressureKernel;
    private int jacobiDiffuseKernel;
    private int bufferToTextureKernel;
    private int AddInkKernel;
    private int removeDivergenceKernel;
    private int userForceKernel;
    private int boundaryKernel;

    //initializer
    public void Init()
    {
        //672 * 672 grid elements, each 4 floats long
        int bufferLength = gridResolution * gridResolution;
        int elementSize = sizeof(float) * 4;
        velocityBuffer = new ComputeBuffer(bufferLength, elementSize);
        divergenceBuffer = new ComputeBuffer(bufferLength, elementSize);
        tempBuffer = new ComputeBuffer(bufferLength, elementSize);
        pressureBuffer = new ComputeBuffer(bufferLength, elementSize);
        inkBuffer = new ComputeBuffer(bufferLength, elementSize);
        outTexture = new RenderTexture(gridResolution, gridResolution, 0)
        {
            //allow shader to write to texture
            enableRandomWrite = true,
        };
        outTexture.Create();
        //set constant shader values to avoid reassigning
        fluidShader.SetInt("_resolution", gridResolution);
        fluidShader.SetFloat("_timeStep", 1.0f);
        Color inkColour = new Color(0.3f, 0.4f, 0.6f, 1.0f);
        fluidShader.SetVector("_dye_colour", inkColour);
        //find kernel indexes
        AddInkKernel = fluidShader.FindKernel("AddInk");
        bufferToTextureKernel = fluidShader.FindKernel("Ink_Buffer_To_Texture");
        copyBufferKernel = fluidShader.FindKernel("Copy_Temp");
        clearBufferKernel = fluidShader.FindKernel("Clear_Buffer");
        advectionKernel = fluidShader.FindKernel("advection");
        divergenceKernel = fluidShader.FindKernel("divergence");
        removeDivergenceKernel = fluidShader.FindKernel("calculate_divergence_free");
        jacobiPressureKernel = fluidShader.FindKernel("Jacobi_Solve_Pressure");
        jacobiDiffuseKernel = fluidShader.FindKernel("Jacobi_Solve_Diffusion");
        userForceKernel = fluidShader.FindKernel("AddForce_mouse");
        boundaryKernel = fluidShader.FindKernel("Boundary");
    }

    public void ReleaseResources()
    {
        //release buffers and texture (garbage collector will do this anyway but Unity will complain)
        outTexture.Release();
        tempBuffer.Release();
        velocityBuffer.Release();
        pressureBuffer.Release();
        divergenceBuffer.Release();
        inkBuffer.Release();
    }

    //Advection causes the fluid to move based on its current motion, for details see shader code
    private void Advect(ComputeBuffer fieldBuffer, float disspationConstant)
    {
        fluidShader.SetFloat("_dissipation_const", disspationConstant);
        Shader.SetGlobalBuffer("_new_advect_field", tempBuffer);
        Shader.SetGlobalBuffer("_curr_advect_field", fieldBuffer);
        Shader.SetGlobalBuffer("_velocity_field", velocityBuffer);
        DispatchKernel(fluidShader, advectionKernel, (uint)gridResolution, (uint)gridResolution, 1);
        //overwrite the buffer with new advected version
        Shader.SetGlobalBuffer("_Temp", tempBuffer);
        Shader.SetGlobalBuffer("_Final", fieldBuffer);
        DispatchKernel(fluidShader, copyBufferKernel, (uint)gridResolution*(uint)gridResolution, 1, 1);
    }

    //Diffusion handels motion that diffuses through the fluid due to viscosity 
    private void Diffuse(ComputeBuffer bufferForDiffusion)
    {
        fluidShader.SetFloat("_viscosity", viscosity);
        bool useTemp = false;
        //we use a jacobi solver to solve a system of linear equations with 5 unknowns, the more iterations, the more accurate the result
        for (int i = 0; i < jacobiIterations; i++)
        {
            //in order to prevent a race condition alternate between using the temp buffer and diffusion buffer, writing to the diffusion buffer in the final solver step
            useTemp = !useTemp;
            if (useTemp)
            {
                Shader.SetGlobalBuffer("_curr_diffusion_buffer", bufferForDiffusion);
                Shader.SetGlobalBuffer("_guess_diffusion_buffer", bufferForDiffusion);
                Shader.SetGlobalBuffer("_final_diffusion_buffer", tempBuffer);
            } else
            {
                Shader.SetGlobalBuffer("_curr_diffusion_buffer", tempBuffer);
                Shader.SetGlobalBuffer("_guess_diffusion_buffer", tempBuffer);
                Shader.SetGlobalBuffer("_final_diffusion_buffer", bufferForDiffusion);
            }
           DispatchKernel(fluidShader, jacobiDiffuseKernel, (uint)gridResolution, (uint)gridResolution, 1);
        }
    }

    //add coloured "ink" to the fluid to visualize movement
    private void Ink()
    {
       Shader.SetGlobalBuffer("_ink_buffer", inkBuffer);
       DispatchKernel(fluidShader, AddInkKernel, (uint)gridResolution, (uint)gridResolution, 1);
    }

    //add force from mouse movement
    private void UserForce() 
    {
       Shader.SetGlobalBuffer("_force_buffer", velocityBuffer);
       DispatchKernel(fluidShader, userForceKernel, (uint)gridResolution, (uint)gridResolution, 1);
    }

    //projection is the final stage, once the field has been advected, diffused and had force added our field is no longer divergence free
    //to solve this we can calculate a pressure field and use this to project onto a field free of divergence 
    private void Project(ComputeBuffer fieldBuffer)
    {
        //work out divergence field
        Shader.SetGlobalBuffer("_vector_field", fieldBuffer);
        Shader.SetGlobalBuffer("_divergence_buffer", divergenceBuffer);
        DispatchKernel(fluidShader, divergenceKernel, (uint)gridResolution, (uint)gridResolution, 1);
        //clear pressure buffer
        Shader.SetGlobalBuffer("_Buffer_To_Clear", pressureBuffer);
        DispatchKernel(fluidShader, clearBufferKernel, (uint)gridResolution * (uint)gridResolution, 1, 1);
        //works in the same maner as diffuse jacobi
        bool useTemp = false;
        for (int i = 0; i < jacobiIterations; i++)
        {
            useTemp = !useTemp;
            if (useTemp)
            {
                CheckEdges(pressureBuffer);
                Shader.SetGlobalBuffer("_guess_pressure_buffer", pressureBuffer);
                Shader.SetGlobalBuffer("_final_pressure_buffer", tempBuffer);
            } 
            else
            {
                CheckEdges(tempBuffer);
                Shader.SetGlobalBuffer("_guess_pressure_buffer",  tempBuffer);
                Shader.SetGlobalBuffer("_final_pressure_buffer", pressureBuffer);
            }
            DispatchKernel(fluidShader, jacobiPressureKernel, (uint)gridResolution, (uint)gridResolution, 1);
        }
        CheckEdges(pressureBuffer);
        //Calculate divergence free field using pressure
        Shader.SetGlobalBuffer("_field_with_divergence", fieldBuffer);
        Shader.SetGlobalBuffer("_pressure", pressureBuffer);
        Shader.SetGlobalBuffer("_final_field", tempBuffer);
        DispatchKernel(fluidShader, removeDivergenceKernel, (uint)gridResolution, (uint)gridResolution, 1);
        //copy from temp
        Shader.SetGlobalBuffer("_Temp", tempBuffer);
        Shader.SetGlobalBuffer("_Final", fieldBuffer);
        DispatchKernel(fluidShader, copyBufferKernel, (uint)gridResolution* (uint)gridResolution, 1, 1);
    }

    //make sure that edge values of the field stay at 0 to prevent unwanted behaviour such as bleeding of dye and velocity from edges
    private void CheckEdges(ComputeBuffer buffer)
    {
       Shader.SetGlobalBuffer("_boundary_field", buffer);
       //this is only executed on edge pixels, 672 on each edge so 672 * 4 total
       DispatchKernel(fluidShader, boundaryKernel, (uint)gridResolution * 4, 1, 1);
    }

    //blit renderTexture to the screen
    public void RenderImage(RenderTexture destination)
    {
        fluidShader.SetTexture(bufferToTextureKernel, "_Out_Texture", outTexture);
        DispatchKernel(fluidShader, bufferToTextureKernel, (uint)gridResolution, (uint)gridResolution, 1);
        Graphics.Blit(outTexture, destination);
    }

    public void Tick()
    {
        //find mouse position in viewport space
        Vector2 mouseLatest  = Camera.main.ScreenToViewportPoint(Input.mousePosition);
        //normalise x and y components then multiply by resolution to get the mouse position in correct coordinates 
        mouseLatest  = new Vector2(Mathf.Clamp(mouseLatest.x, 0, 1), Mathf.Clamp(mouseLatest.y, 0, 1));
        mouseLatest =  new Vector2(mouseLatest.x * gridResolution, mouseLatest.y * gridResolution);
        fluidShader.SetVector("_mouse_latest",    mouseLatest);   
        fluidShader.SetVector("_mouse_old",    oldMousePos); 
        oldMousePos = mouseLatest;

        //CONTROL LOOP
        //ADVECT
        Advect(inkBuffer, inkDissipation);
        Advect(velocityBuffer, velocityDissipation);
        //DIFFUSE
        Diffuse (inkBuffer);
        Diffuse (velocityBuffer);
        CheckEdges(inkBuffer);
        CheckEdges(velocityBuffer);
        //FORCE
        UserForce();
        Ink();
        //PROJECT
        //projecting ink is not necessary but gives a visually appealing effect
        Project(inkBuffer);
        Project(velocityBuffer);

        Shader.SetGlobalBuffer("_Ink_Buffer", inkBuffer);
    }

    private static void  DispatchKernel(ComputeShader shader, int handle, uint threadX, uint threadY, uint threadZ)
    {
        // if resolution is 600 for example a usual desired thread num would be 600 600 1
        uint groupSizeX, groupSizeY, groupSizeZ;
        //Get the number of threads in a group (eg. how many ALU's per control unit)
        //lets assume 16 16 1 (sometimes 256 1 1)
        shader.GetKernelThreadGroupSizes(handle, out groupSizeX, out groupSizeY, out groupSizeZ);
        //ints round to get a integrer position
        int finalThreadX = (int) (threadX / groupSizeX);
        int finalThreadY = (int) (threadY / groupSizeY);
        int finalThreadZ = (int) (threadZ / groupSizeZ);
        //600/16 , 600/16, 1/1 = 37, 37 ,1
        //will dispatch with something like 37, 37 , 1
        //this essentially means with a kernel size of 16*16, there are 37 *37 different spots the kernel can be in. the entire pixel image is split up into a 37 * 37 grid each of which is operated on by a 16 * 16 kernel
        shader.Dispatch(handle, finalThreadX, finalThreadY, finalThreadZ);
    }
}

