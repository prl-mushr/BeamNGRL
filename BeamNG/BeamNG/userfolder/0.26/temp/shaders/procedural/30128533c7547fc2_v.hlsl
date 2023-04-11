//*****************************************************************************
// BNG -- HLSL procedural shader
//*****************************************************************************

#include "shaders/common/bng.hlsl"
// Dependencies:
#include "shaders/common/lighting.hlsl"
#include "shaders/common/lighting/shadowMap/shadowMapPSSM.h.hlsl"
#include "shaders/common/bng.hlsl"

// Features:
// Vert Position
// Vert Normal
// Material input 0
// Diffuse Color 0
// Specular Data 0
// MaterialLayerBlendHLSL 0
// MaterialOutputHLSL 0
// Deferred RT Lighting
// Visibility
// HDR Output
// Final color Output
// MFT_MaterialDeprecated

struct VertData
{
   float3 position        : POSITION;
   float3 normal          : NORMAL;
   float3 T               : TANGENT;
   float3 B               : BINORMAL;
   float2 texCoord        : TEXCOORD0;
   float2 texCoord2       : TEXCOORD1;
};


struct ConnectData
{
   float4 hpos            : SV_Position;
   float3 vNormal         : TEXCOORD0;
   float4 out_texCoord    : TEXCOORD1;
   float4 screenspacePos  : TEXCOORD2;
};


//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
cbuffer cspMaterial : REGISTER(b4, space1) 
{
    uniform float    roughnessFactor0;
    uniform float4   diffuseMaterialColor0;
    uniform float4   specularColor0 ;
}

ConnectData main( VertData IN, uint svInstanceID : SV_InstanceID, uint vertexID : SV_VertexID, [[vk::builtin("BaseInstance")]] uint svInstanceBaseID : A
)
{
   ConnectData OUT = (ConnectData)0;

   // Vert Position
   float4x4 objTrans = uObjectTrans;
   float4x4 worldViewOnly = mul( worldToCamera, objTrans ); // Instancing!
   OUT.hpos = mul(mul(cameraToScreen, worldViewOnly), float4(IN.position.xyz,1));
   
   // Vert Normal
   OUT.vNormal = mul(objTrans, float4( normalize(IN.normal), 0.0 ) ).xyz;
   
   // Material input 0
   
   // Diffuse Color 0
   
   // Specular Data 0
   
   // MaterialLayerBlendHLSL 0
   
   // MaterialOutputHLSL 0
   OUT.out_texCoord = float4(IN.texCoord.xy, IN.texCoord2.xy);
   
   // Deferred RT Lighting
   OUT.screenspacePos = OUT.hpos;
   
   // Visibility
   
   // HDR Output
   
   // Final color Output
   
   // MFT_MaterialDeprecated
   
   return OUT;
}
