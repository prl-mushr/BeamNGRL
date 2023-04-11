//*****************************************************************************
// BNG -- HLSL procedural shader
//*****************************************************************************

#include "shaders/common/bng.hlsl"
// Dependencies:
#include "shaders/common/bng.hlsl"
#include "shaders/common/lighting.hlsl"

// Features:
// Vert Position
// Vert Normal
// Material input 0
// Specular Data 0
// MaterialLayerBlendHLSL 0
// MaterialOutputHLSL 0
// Visibility
// Eye Space Depth (Out)
// GBuffer Conditioner
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
   float4 wsEyeVec        : TEXCOORD2;
};


//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
cbuffer cspMaterial : REGISTER(b4, space1) 
{
    uniform float    roughnessFactor0;
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
   
   // Specular Data 0
   
   // MaterialLayerBlendHLSL 0
   
   // MaterialOutputHLSL 0
   OUT.out_texCoord = float4(IN.texCoord.xy, IN.texCoord2.xy);
   
   // Visibility
   
   // Eye Space Depth (Out)
   float3 wsPosition = mul( objTrans, float4( IN.position.xyz, 1 ) ).xyz;
   OUT.wsEyeVec = float4( wsPosition.xyz - eyePosWorld, 1 );
   
   // GBuffer Conditioner
   
   // MFT_MaterialDeprecated
   
   return OUT;
}
