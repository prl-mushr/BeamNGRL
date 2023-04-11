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

struct ConnectData
{
   float4 vpos            : SV_Position;
   float3 vNormal         : TEXCOORD0;
   float4 texCoord        : TEXCOORD1;
   float4 wsEyeVec        : TEXCOORD2;
};


struct Fragout
{
   float4 target0 : SV_Target0;
};


//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------


cbuffer cspMaterial : REGISTER(b4, space1) 
{
    uniform float    roughnessFactor0;
    uniform float4   specularColor0;
}

Fragout main( ConnectData IN, bool isFrontFacing : SV_IsFrontFace
)
{
   Fragout OUT = (Fragout)0;

   // Vert Position
   
   // Vert Normal
    float3 layerNormal = float3(0, 0, 1);
    layerNormal = normalize(IN.vNormal);
   
   // Material input 0
    float layerRoughness = 1.0f;
   layerRoughness = roughnessFactor0;
   
   // Specular Data 0
    float4 layerSpecularColor = float4(1, 1, 1, 1);
    layerSpecularColor = specularColor0;
   
   // MaterialLayerBlendHLSL 0
    float4 materialDiffuseColor = float4(1, 1, 1, 1);
    float4 layerDiffuseColor = float4(1, 1, 1, 1);
   materialDiffuseColor = lerp(materialDiffuseColor, saturate(layerDiffuseColor), 1);
    float4 materiaSpecularColor = float4(1, 1, 1, 1);
   materiaSpecularColor = lerp(materiaSpecularColor, saturate(layerSpecularColor), 1);
    float3 materialNormal = float3(0, 0, 1);
   materialNormal = normalize(lerp(materialNormal, normalize(layerNormal), 1));
    float4 materialReflectivity = 1;
    float4 layerReflectivity = 1;
   materialReflectivity = lerp(materialReflectivity, saturate(layerReflectivity), 1);
    float materialDetailFactor = 1;
    float layerDetailFactor = 1;
   materialDetailFactor = lerp(materialDetailFactor, saturate(layerDetailFactor), 1);
    float materialRoughness = 1.0f;
   materialRoughness = lerp(materialRoughness, saturate(layerRoughness), 1);
    float materialAO = 1.0f;
    float layerAO = 1.0f;
   materialAO = lerp(materialAO, saturate(layerAO), 1);
    float materialClearCoat = 1.0f;
    float layerClearCoat = 1.0f;
   materialClearCoat = lerp(materialClearCoat, saturate(layerClearCoat), 1);
    float materialClearCoatRoughness = 1.0f;
    float layerClearCoatRoughness = 1.0f;
   materialClearCoatRoughness = lerp(materialClearCoatRoughness, saturate(layerClearCoatRoughness), 1);
    float3 materialClearCoat2ndNormal = materialNormal;
    float3 layerClearCoat2ndNormal = float3(0, 0, 1);
   materialClearCoat2ndNormal = lerp(materialClearCoat2ndNormal, saturate(layerClearCoat2ndNormal), 1);
    float materialMetalness = 1.0f;
    float layerMetalness = 1.0f;
   materialMetalness = lerp(materialMetalness, saturate(layerMetalness), 1);
   layerDiffuseColor = 1;
   layerMetalness = 1;
   layerSpecularColor = 1;
   layerNormal = float3(0, 0, 1);
   layerReflectivity = 1;
   layerDetailFactor = 1;
   layerRoughness = 1;
   layerAO = 1;
   layerClearCoat = 1;
   layerClearCoatRoughness = 1;
   layerClearCoat2ndNormal = float3(0, 0, 1);
   
   // MaterialOutputHLSL 0
   materialClearCoat2ndNormal = materialNormal;
   Material material = (Material)0;
   material.baseColor = materialDiffuseColor.rgb;
   material.opacity = materialDiffuseColor.a;
   material.metallic = materialMetalness;
   material.roughness = materialRoughness;
   material.ambientOclussion = materialAO;
   material.clearCoat = materialClearCoat;
   material.clearCoatRoughness = materialClearCoatRoughness;
    float3 materialEmmisive = 0.0f;
   material.emmisive = materialEmmisive;
   material.layerCount = 1;
   material.texCoords = IN.texCoord;
   material.flags |= MaterialFlagDeprecated;
   materialDiffuseColor = (materialDiffuseColor);
   materiaSpecularColor = (max(float4(materiaSpecularColor.rgb, 1), 0.04));
   
   // Visibility
   float visibility = 1;
   visibility *= uVisibility;
   fizzle( IN.vpos.xy, visibility );
   
   // Eye Space Depth (Out)
#ifndef CUBE_SHADOW_MAP
   float eyeSpaceDepth = dot(vEye, (IN.wsEyeVec.xyz / IN.wsEyeVec.w));
#else
   float eyeSpaceDepth = length( IN.wsEyeVec.xyz / IN.wsEyeVec.w ) * oneOverFarplane.x;
#endif
   
   // GBuffer Conditioner
    float4 normal_depth;    float4 out_spec_none;
   encodeGBuffer(materialNormal, materialRoughness, materiaSpecularColor.rgb, materialDiffuseColor.a, normal_depth, out_spec_none);

   // generic conditioner: no conditioning performed
   OUT.target0 = normal_depth;
   
   // MFT_MaterialDeprecated
   

   return OUT;
}
