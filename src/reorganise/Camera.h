#pragma once

#include <sutil/vec_math.h>
#include <cmath>

using ::length;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class Camera
{
public:
    float3 position;
    float3 lookAt;
    float3 up;
    float fov;
    float aspectRatio;

    float3 target;
    float distance;
    float yaw;
    float pitch;
    float minPitch;
    float maxPitch;
    float moveSpeed;

private:
    float3 u, v, w;

    void updateBasis();
    void updatePositionFromOrbit();

public:
    void setPosition(const float3 &pos);
    void setLookAt(const float3 &target);

    Camera();
    Camera(const float3 &pos, const float3 &target, const float3 &upVec, float fovDegrees, float aspect);

    void setAspectRatio(float aspect);

    void orbit(float deltaYaw, float deltaPitch);

    void dolly(float delta);

    void pan(float deltaX, float deltaY);

    void setTarget(const float3 &newTarget);

    void setMoveSpeed(float speed);

    const float3 &getPosition() const { return position; }
    const float3 &getLookAt() const { return lookAt; }
    float getAspectRatio() const { return aspectRatio; }

    float3 getU() const;
    float3 getV() const;
    float3 getW() const;
};
