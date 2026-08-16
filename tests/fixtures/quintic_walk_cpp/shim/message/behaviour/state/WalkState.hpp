#ifndef SHIM_MESSAGE_BEHAVIOUR_STATE_WALKSTATE_HPP
#define SHIM_MESSAGE_BEHAVIOUR_STATE_WALKSTATE_HPP

// Minimal stand-in for the protobuf-generated WalkState. Enum values match
// shared/message/behaviour/state/WalkState.proto exactly. State mimics the
// generated wrapper struct, which exposes the enumerator through `.value`.

namespace message::behaviour::state {

    struct WalkState {
        struct State {
            enum Value { UNKNOWN = 0, STARTING = 1, WALKING = 2, STOPPING = 3, STOPPED = 4 };
            Value value = UNKNOWN;
            State() = default;
            State(Value v) : value(v) {}
            operator Value() const {
                return value;
            }
        };

        enum class Phase { DOUBLE = 0, LEFT = 1, RIGHT = 2 };
    };

}  // namespace message::behaviour::state

#endif
