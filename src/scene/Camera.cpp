

#include "Camera.h"

void Camera::updateBasis()
{
    w = normalize(lookAt - position);

    
    u = normalize(make_float3(w.z, 0, -w.x));

    
    v = make_float3(
        w.y * u.z - w.z * 0.0f,
        w.z * u.x - w.x * u.z,
        w.x * 0.0f - w.y * u.x);

    
    if (length(v) < 1e-6f)
    {
        v = make_float3(0, 1, 0);
    }
    else
    {
        v = normalize(v);
    }

    
    if (v.y < 0)
    {
        v = v * -1.0f;
    }
}

void Camera::updatePositionFromOrbit()
{
    
    pitch = fmaxf(minPitch, fminf(maxPitch, pitch));

    
    float cosPitch = cosf(pitch);
    float sinPitch = sinf(pitch);
    float cosYaw = cosf(yaw);
    float sinYaw = sinf(yaw);

    position.x = target.x + distance * cosPitch * cosYaw;
    position.y = target.y + distance * sinPitch;
    position.z = target.z + distance * cosPitch * sinYaw;

    lookAt = target;
    updateBasis();
}

void Camera::setPosition(const float3& pos)
{
    position = pos;
    updateBasis();
}

void Camera::setLookAt(const float3& target)
{
    lookAt = target;
    updateBasis();
}

Camera::Camera()
    : position(make_float3(0, 0, 5)), lookAt(make_float3(0, 0, 0)), up(make_float3(0, 1, 0)), fov(45.0f * M_PI / 180.0f), aspectRatio(1.0f),
      target(make_float3(0, 0, 0)), distance(5.0f), yaw(0.0f), pitch(0.0f), minPitch(-M_PI/2.0f + 0.1f), maxPitch(M_PI/2.0f - 0.1f), moveSpeed(1.0f)
{
    updateBasis();
}

Camera::Camera(const float3 &pos, const float3 &target, const float3 &upVec, float fovDegrees, float aspect)
    : position(pos), lookAt(target), up(upVec), fov(fovDegrees * M_PI / 180.0f), aspectRatio(aspect),
      target(target), distance(length(pos - target)), yaw(0.0f), pitch(0.0f), minPitch(-M_PI/2.0f + 0.1f), maxPitch(M_PI/2.0f - 0.1f), moveSpeed(1.0f)
{
    
    float3 dir = normalize(pos - target);
    yaw = atan2f(dir.z, dir.x);
    pitch = asinf(dir.y);
    updateBasis();
}

void Camera::setAspectRatio(float aspect)
{
    aspectRatio = aspect;
}


void Camera::orbit(float deltaYaw, float deltaPitch)
{
    yaw += deltaYaw * moveSpeed * 0.01f;
    pitch += deltaPitch * moveSpeed * 0.01f;
    updatePositionFromOrbit();
}


void Camera::dolly(float delta)
{
    distance *= (1.0f - delta * moveSpeed * 0.001f);
    distance = fmaxf(0.1f, distance); 
    updatePositionFromOrbit();
}


void Camera::pan(float deltaX, float deltaY)
{
    
    float3 right = u;
    float3 up = v;

    
    float panScale = distance * tanf(fov * 0.5f) * moveSpeed * 0.0001f;

    target += right * (-deltaX * panScale) + up * (deltaY * panScale);
    updatePositionFromOrbit();
}


void Camera::setTarget(const float3& newTarget)
{
    target = newTarget;
    updatePositionFromOrbit();
}


void Camera::setMoveSpeed(float speed)
{
    moveSpeed = speed;
}


float3 Camera::getU() const
{
    
    float wlen = length(lookAt - position);
    float vlen = wlen * tanf(0.5f * fov);
    float ulen = vlen * aspectRatio;
    return u * ulen;
}

float3 Camera::getV() const
{
    
    float wlen = length(lookAt - position);
    float vlen = wlen * tanf(0.5f * fov);
    return v * vlen;
}

float3 Camera::getW() const
{
    
    return lookAt - position;
}



