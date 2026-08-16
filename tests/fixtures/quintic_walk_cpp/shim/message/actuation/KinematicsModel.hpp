#ifndef SHIM_MESSAGE_ACTUATION_KINEMATICSMODEL_HPP
#define SHIM_MESSAGE_ACTUATION_KINEMATICSMODEL_HPP

// Minimal stand-in for the protobuf-generated KinematicsModel, carrying only
// the fields calculate_leg_joints reads. Field types match the .proto exactly
// (float / int32) so the arithmetic matches the real thing bit for bit.

namespace message::actuation {

    struct KinematicsModel {
        struct Leg {
            double HIP_OFFSET_X            = 0.0f;
            double HIP_OFFSET_Y            = 0.0f;
            double HIP_OFFSET_Z            = 0.0f;
            double UPPER_LEG_LENGTH        = 0.0f;
            double LOWER_LEG_LENGTH        = 0.0f;
            double FOOT_HEIGHT             = 0.0f;
            double LENGTH_BETWEEN_LEGS     = 0.0f;
            int LEFT_TO_RIGHT_HIP_YAW     = 1;
            int LEFT_TO_RIGHT_HIP_ROLL    = 1;
            int LEFT_TO_RIGHT_HIP_PITCH   = 1;
            int LEFT_TO_RIGHT_KNEE        = 1;
            int LEFT_TO_RIGHT_ANKLE_PITCH = 1;
            int LEFT_TO_RIGHT_ANKLE_ROLL  = 1;
        } leg;
    };

}  // namespace message::actuation

#endif
